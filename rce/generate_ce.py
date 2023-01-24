import numpy as np
import pandas as pd
import os
import time

# optimization
from pyomo import environ
from pyomo.environ import *

# utils
import rce


def write_tables(save_path, X_train, y_train, clf, clf_type, task_type):
    """
        This function uses constraint_learning to write the ML models in
        a tabular format and it saves the csv file in 'save_path'.
    """
    if not os.path.exists(save_path + '/' + clf_type):
        os.makedirs(save_path + '/' + clf_type)
    constraintL = rce.ConstraintLearning(X_train, y_train, clf, clf_type)
    constraint_add = constraintL.constraint_extrapolation(task_type)
    constraint_add.to_csv(save_path + '/%s/model.csv' % (clf_type), index=False)
    print(f'{clf_type} tables saved.')


def base_model(u, F, F_b, F_int, F_coh, I, L, P):
    """
        This function returns a model with base constraints (coherence, immutability, etc.)
        necessary to generate counterfactual explanations.
    """
    model = ConcreteModel('RCE')
    model.x = Var(F, domain=Reals, name=['ce_%s' % str(ce) for ce in F])  # counterfactual features

    for i in F_b:
        model.x[i].domain = Binary

    for i in F_int:
        model.x[i].domain = NonNegativeIntegers

    for cat in F_coh.keys():
        model.add_component('coherence_' + cat, Constraint(expr=sum(model.x[i] for i in F_coh[cat]) == 1))

    def immutability(model, i):
        # print()
        return model.x[i] == u[i].item()

    model.add_component('immutability', Constraint(I, rule=immutability))

    def larger(model, i):
        return model.x[i] >= u[i].item()

    model.add_component('larger_than', Constraint(L, rule=larger))

    def positive(model, i):
        return model.x[i] >= 0

    model.add_component('positive', Constraint(P, rule=positive))

    return model


def find_maxrad(x_, clf_type, save_path, F, F_b, F_int, F_coh, I, L, P, class_, obj_type):
    """
        This function finds the closest solution to the (robust) counterfactual explanation 'x_'
        that is classified as 'class_', namely the class of the factual instance.
    """
    clf_path = save_path + '/' + clf_type + '/model.csv'
    mfile = pd.read_csv(clf_path)
    model = base_model(x_, F, F_b, F_int, F_coh, I, L, P)

    # Define the objective function. It must be the same as the uncertainty set
    # used for the robust counterfactual explanation
    model = add_objective(model, x_, obj_type, maxrad=True)
    # Upper and lower bounds for the validity constraints
    lb, ub = None, None
    if clf_type in ['cart', 'gbm', 'rf']:
        if class_ == 1:
            lb, ub = 0.499999, None
        else:
            lb, ub = None, 0.500001
    elif clf_type in ['linear', 'svm', 'mlp']:
        if class_ == 1:
            lb, ub = 0., None
        else:
            lb, ub = None, 0

    # Definition of variables for the validity constraints
    model.y = Var(Any, dense=False, domain=Reals)
    model.l = Var(Any, dense=False, domain=Binary)
    model.y_viol = Var(Any, dense=False, domain=Binary)
    model.v = Var(Any, dense=False, domain=NonNegativeReals)
    model.v_ind = Var(Any, dense=False, domain=Binary)
    model.lam = Var(Any, dense=False, domain=Reals, bounds=(0, 1))

    # Adding the validity constraints
    if clf_type in ['linear', 'svm']:
        model = constraints_linear(model, 'prediction', class_, mfile, F, 0, None, lb, ub)
    elif clf_type == 'cart':
        model = constraints_tree(model, 'prediction', mfile, F, 0, None, lb, ub,
                                 adversarial_algorithm_bool=False, S=[], adv_prob=False)  # check it
    elif clf_type == 'rf':
        model = constraints_rf(model, 'prediction', mfile, F, 0, None, lb, ub,
                               adversarial_algorithm_bool=False, S=[], adv_prob=False)
    elif clf_type == 'gbm':
        model = constraints_gbm(model, 'prediction', mfile, F, 0, None, lb, ub,
                               adversarial_algorithm_bool=False, S=[], adv_prob=False)
    elif clf_type == 'mlp':
        model = constraints_mlp(model, 'prediction', mfile, F, lb, ub, adv_prob=True, S=[])  # check it

    # Solving the optimization model
    opt = SolverFactory('gurobi')
    opt.solve(model)
    results = opt.solve(model)
    sol = [value(model.x[i]) for i in x_.columns]
    distance = value(model.OBJ)

    # print(value(model.y['prediction']))

    if obj_type == 'l2':
        distance = np.sqrt(distance)

    return sol, np.round(distance, 5)


def constraints_linear(model, outcome, class_, linear_table, F, rho, unc_type, lb, ub):
    """
        This function generates the constraints for a linear predictive model.
    """
    intercept = linear_table['intercept'][0]
    coeff = linear_table.drop(['intercept'], axis=1, inplace=False).loc[0, :]
    if class_ == 1:
        sign = 1
    else:
        sign = -1
    if unc_type == 'l2':
        robust_term = sign * rho * np.linalg.norm(coeff, 2)
    elif unc_type == 'linf':
        robust_term = sign * rho * np.linalg.norm(coeff, 1)
    else:
        robust_term = 0

    model.add_component('linear_model', Constraint(
        expr=model.y[outcome] == sum(model.x[i] * coeff.loc[i] for i in F) + intercept + robust_term))

    if not pd.isna(ub):
        model.add_component('ub', Constraint(expr=model.y['prediction'] <= ub))
    if not pd.isna(lb):
        model.add_component('lb', Constraint(expr=model.y['prediction'] >= lb))

    return model


def constraints_tree(model, outcome, tree_table, F, rho, unc_type, lb, ub, adversarial_algorithm_bool, S, adv_prob):
    """
        This function generates the constraints for a decision tree and each decision tree of random forest and
        gradient boosting.
    """
    M = 1000
    leaf_values = tree_table.loc[:, ['ID', 'prediction']].drop_duplicates().set_index('ID')
    # Row-level information:
    intercept = tree_table['threshold']
    coeff = tree_table.drop(['ID', 'threshold', 'prediction', 'node_ID'], axis=1, inplace=False).reset_index(
        drop=True)
    l_ids = tree_table['ID']
    node_ids = tree_table['node_ID']
    n_constr = coeff.shape[0]
    L = np.unique(tree_table['ID'])

    if not adversarial_algorithm_bool:  # the robust counterfactual explanations will be in one of the leaves
        def constraintsTree_1(model, j):
            if unc_type == 'l2':
                robust_term = rho * np.linalg.norm(coeff.loc[j, :], 2)
            elif unc_type == 'linf':
                robust_term = rho * np.linalg.norm(coeff.loc[j, :], 1)
            elif unc_type == 'l1':
                print('not supported yet!')
            elif pd.isna(unc_type):
                robust_term = 0
            return sum(model.x[i] * coeff.loc[j, i] for i in F) + robust_term <= intercept.iloc[j] + M * (
                    1 - model.l[(outcome, str(l_ids.iloc[j]))])

        model.add_component(outcome + '_splits', Constraint(range(n_constr), rule=constraintsTree_1))

        def constraintsTree_2(model):
            return sum(model.l[(outcome, str(i))] for i in L) == 1

        model.add_component(outcome + '_oneleaf', Constraint(rule=constraintsTree_2))

        def constraintTree_3(model):
            return model.y[outcome] == sum(leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i))] for i in L)

        model.add_component('DT_' + outcome, Constraint(rule=constraintTree_3))

        if not pd.isna(ub):
            model.add_component('ub', Constraint(expr=model.y[outcome] <= ub))
        if not pd.isna(lb):
            model.add_component('lb', Constraint(expr=model.y[outcome] >= lb))
    else:  # adversarial algorithm
        if adv_prob:  # adversarial problem
            def constraintsTree_1(model, j):
                return sum(model.x[i] * coeff.loc[j, i] for i in F) + model.w[(outcome, node_ids[j])] <= intercept.iloc[
                    j] + M * (1 - model.l[(outcome, str(l_ids.iloc[j]))])
            model.add_component(outcome + '_splits', Constraint(range(n_constr), rule=constraintsTree_1))
            def constraintsTree_2(model):
                return sum(model.l[(outcome, str(i))] for i in L) == 1
            model.add_component(outcome + '_oneleaf', Constraint(rule=constraintsTree_2))
            def constraintTree_3(model):
                return model.y[outcome] == sum(
                    leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i))] for i in L)

            model.add_component('DT' + outcome, Constraint(rule=constraintTree_3))
            #
            # def constraintsTree_4(model, i):
            #     print(M)
            #     print(tree_table[tree_table['node_ID'] == i]['ID'])
            #     return model.w[(outcome, i)] <= M * sum(
            #         model.l[(outcome, str(j))] for j in tree_table[tree_table['node_ID'] == i]['ID'])
            def constraintsTree_5(model, i):
                return model.mu[(outcome)] <= model.w[(outcome, i)]
            for i in np.unique(node_ids):
                # model.add_component(outcome + f'contr_4_{i}', Constraint([i], rule=constraintsTree_4))
                model.add_component(outcome + f'contr_5_{i}', Constraint([i], rule=constraintsTree_5))

            if not pd.isna(ub):
                model.add_component('ub', Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                model.add_component('lb', Constraint(expr=model.y[outcome] >= lb))

        else:
            def constraintsTree_1(model, j, s):
                return sum((model.x[i] + S[s][i]) * coeff.loc[j, i] for i in F) <= intercept.iloc[j] + M * (
                        1 - model.l[(outcome, str(l_ids.iloc[j]), s)])
            model.add_component(outcome + '_splits',
                                Constraint(range(n_constr), range(len(S)), rule=constraintsTree_1))

            def constraintsTree_2(model, s):
                return sum(model.l[(outcome, str(i), s)] for i in L) == 1

            model.add_component(outcome + '_oneleaf', Constraint(range(len(S)), rule=constraintsTree_2))

            def constraintTree_3(model, s):
                return model.y[(outcome, s)] == sum(
                    leaf_values.loc[i, 'prediction'] * model.l[(outcome, str(i), s)] for i in L)

            model.add_component('DT' + outcome, Constraint(range(len(S)), rule=constraintTree_3))
            if not pd.isna(ub):
                def constr_ub(model, s):
                    return model.y[(outcome, s)] <= ub

                model.add_component('ub', Constraint(range(len(S)), rule=constr_ub))

            if not pd.isna(lb):
                def constr_lb(model, s):
                    return model.y[(outcome, s)] >= lb

                model.add_component('lb', Constraint(range(len(S)), rule=constr_lb))

    return model


def constraints_rf(model, outcome, forest_table, F, rho, unc_type, lb, ub, adversarial_algorithm_bool, S, adv_prob):
    """
        This function generates the constraints for a random forest.
    """
    forest_table_temp = forest_table.copy()
    forest_table_temp['Tree_id'] = [outcome + '_' + str(i) for i in forest_table_temp['Tree_id']]
    T = np.unique(forest_table_temp['Tree_id'])
    if not adversarial_algorithm_bool:
        for i, t in enumerate(T):
            tree_table = forest_table_temp.loc[forest_table_temp['Tree_id'] == t, :].drop('Tree_id', axis=1)
            constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                             adversarial_algorithm_bool=False, S=[], adv_prob=False)

        model.add_component('RF' + outcome,
                            Constraint(rule=model.y[outcome] == 1 / len(T) * quicksum(model.y[j] for j in T)))
        if not pd.isna(ub):
            model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
        if not pd.isna(lb):
            model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
    else:
        if adv_prob:
            for i, t in enumerate(T):
                tree_table = forest_table_temp.loc[forest_table_temp['Tree_id'] == t, :].drop(
                    'Tree_id', axis=1).reset_index(drop=True, inplace=False)
                constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                                 adversarial_algorithm_bool=True, S=S, adv_prob=True)
            model.add_component('RF' + outcome,
                                Constraint(rule=model.y[outcome] == 1 / len(T) * quicksum(model.y[j] for j in T)))

            def aux_mu(model, i):
                return model.mu_e[outcome] <= model.mu[i]
            model.add_component('RF_aux_mu', Constraint(T, rule=aux_mu))

            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
        else:
            for i, t in enumerate(T):
                tree_table = forest_table_temp.loc[forest_table_temp['Tree_id'] == t, :].drop(
                    'Tree_id', axis=1).reset_index(drop=True, inplace=False)
                constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                                 adversarial_algorithm_bool=True, S=S, adv_prob=False)

            def constr_RF_output(model, s):
                return model.y[(outcome, s)] == 1 / len(T) * quicksum(model.y[(j, s)] for j in T)
            model.add_component('RF' + outcome, Constraint(range(len(S)), rule=constr_RF_output))

            if not pd.isna(ub):
                def constr_ub(model, s):
                    return model.y[(outcome, s)] <= ub
                model.add_component('ub', Constraint(range(len(S)), rule=constr_ub))
            if not pd.isna(lb):
                def constr_lb(model, s):
                    return model.y[(outcome, s)] >= lb
                model.add_component('lb', Constraint(range(len(S)), rule=constr_lb))

    return model


def constraints_gbm(model, outcome, gbm_table, F, rho, unc_type, lb, ub, adversarial_algorithm_bool, S, adv_prob):
    """
        This function generates the constraints for a gradient boosting.
    """
    gbm_table_temp = gbm_table.copy()
    gbm_table_temp['Tree_id'] = [outcome + '_' + str(i) for i in gbm_table_temp['Tree_id']]
    T = np.unique(gbm_table_temp['Tree_id'])
    if not adversarial_algorithm_bool:
        for i, t in enumerate(T):
            tree_table = gbm_table_temp.loc[gbm_table_temp['Tree_id'] == t, :].drop(['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False)
            constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                             adversarial_algorithm_bool=False, S=[], adv_prob=False)

        def constraint_gbm(model):
            return model.y[outcome] == np.unique(gbm_table_temp['initial_prediction']).item() + np.unique(gbm_table_temp['learning_rate']).item() * quicksum(model.y[j] for j in T)

        model.add_component('GBM'+outcome, Constraint(rule=constraint_gbm))
        if not pd.isna(ub):
            model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
        if not pd.isna(lb):
            model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
    else:
        if adv_prob:
            for i, t in enumerate(T):
                tree_table = gbm_table_temp.loc[gbm_table_temp['Tree_id'] == t, :].drop(['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False).reset_index(drop=True, inplace=False)
                constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                                 adversarial_algorithm_bool=True, S=S, adv_prob=True)

            def constraint_gbm(model):
                return model.y[outcome] == np.unique(gbm_table_temp['initial_prediction']).item() + np.unique(gbm_table_temp['learning_rate']).item() * quicksum(model.y[j] for j in T)

            model.add_component('GBM' + outcome, Constraint(rule=constraint_gbm))

            def aux_mu(model, i):
                return model.mu_e[outcome] <= model.mu[i]
            model.add_component('GBM_aux_mu', Constraint(T, rule=aux_mu))

            if not pd.isna(ub):
                model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
            if not pd.isna(lb):
                model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
        else:
            for i, t in enumerate(T):
                tree_table = gbm_table_temp.loc[gbm_table_temp['Tree_id'] == t, :].drop(['Tree_id', 'initial_prediction', 'learning_rate'], axis=1, inplace=False).reset_index(drop=True, inplace=False)
                constraints_tree(model, t, tree_table, F, rho, unc_type, lb=None, ub=None,
                                 adversarial_algorithm_bool=True, S=S, adv_prob=False)

            def constraint_gbm(model, s):
                return model.y[(outcome, s)] == np.unique(gbm_table_temp['initial_prediction']).item() + np.unique(gbm_table_temp['learning_rate']).item() * quicksum(model.y[(j, s)] for j in T)

            model.add_component('GBM' + outcome, Constraint(range(len(S)), rule=constraint_gbm))

            if not pd.isna(ub):
                def constr_ub(model, s):
                    return model.y[(outcome, s)] <= ub
                model.add_component('ub', Constraint(range(len(S)), rule=constr_ub))
            if not pd.isna(lb):
                def constr_lb(model, s):
                    return model.y[(outcome, s)] >= lb
                model.add_component('lb', Constraint(range(len(S)), rule=constr_lb))

    return model


def constraints_mlp(model, outcome, weights, F, lb, ub, adv_prob, S, M_l=-1e2, M_u=1e2):
    """
        This function generates the constraints for a neural network.
    """
    if adv_prob:
        nodes_input = range(len(F))
        v_input = [model.x[i] for i in F]
        max_layer = max(weights['layer'])
        for l in range(max_layer + 1):
            df_layer = weights.query('layer == %d' % l)
            max_nodes = [k for k in df_layer.columns if 'node_' in k]
            coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
            intercepts_layer = np.array(df_layer['intercept'])
            nodes = df_layer['node']

            if l == max_layer:
                node = nodes.iloc[0]  # only one node in last layer
                model.add_component('MLP' + outcome, Constraint(
                    rule=model.y[outcome] == sum(v_input[i] * coeffs_layer[node, i] for i in nodes_input) +
                         intercepts_layer[node]
                )
                                    )
            else:
                # Save v_pos for input to next layer
                v_pos_list = []
                for node in nodes:
                    ## Initialize variables
                    v_pos_list.append(model.v[(outcome, l, node)])
                    model.add_component('constraint_1_' + str(l) + '_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] >= sum(
                                            v_input[i] * coeffs_layer[node, i] for i in nodes_input) +
                                                        intercepts_layer[
                                                            node]
                                                   )
                                        )
                    model.add_component('constraint_2_' + str(l) + '_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= M_u * (
                                            model.v_ind[(outcome, l, node)])))
                    model.add_component('constraint_3_' + str(l) + '_' + str(node) + outcome,
                                        Constraint(rule=model.v[(outcome, l, node)] <= sum(
                                            v_input[i] * coeffs_layer[node, i] for i in nodes_input) +
                                                        intercepts_layer[
                                                            node] - M_l * (1 - model.v_ind[(outcome, l, node)])
                                                   )
                                        )

                ## Prepare nodes_input for next layer
                nodes_input = nodes
                v_input = v_pos_list

        if not pd.isna(ub):
            model.add_component('ub_' + outcome, Constraint(expr=model.y[outcome] <= ub))
        if not pd.isna(lb):
            model.add_component('lb_' + outcome, Constraint(expr=model.y[outcome] >= lb))
    else:
        nodes_input = range(len(F))
        v_input = [[model.x[i] + S[s][i] for i in F] for s in range(len(S))]
        max_layer = max(weights['layer'])
        for l in range(max_layer + 1):
            df_layer = weights.query('layer == %d' % l)
            max_nodes = [k for k in df_layer.columns if 'node_' in k]
            coeffs_layer = np.array(df_layer.loc[:, max_nodes].dropna(axis=1))
            intercepts_layer = np.array(df_layer['intercept'])
            nodes = df_layer['node']
            if l == max_layer:
                node = nodes.iloc[0]  # only one node in last layer
                for s in range(len(S)):
                    model.add_component('MLP' + outcome + '_' + str(s),
                                        Constraint(rule=model.y[(outcome, s)] == sum(
                                            v_input[s][i] * coeffs_layer[node, i] for i in nodes_input) +
                                                        intercepts_layer[node]
                                                   )
                                        )
            else:
                # Save v_pos for input to next layer
                v_pos_list = []
                for s in range(len(S)):
                    v_pos_list_s = []
                    for node in nodes:
                        ## Initialize variables
                        v_pos_list_s.append(model.v[(outcome, l, node), s])
                        model.add_component('constraint_1_' + str(l) + '_' + str(node) + outcome + '_' + str(s),
                                            Constraint(rule=model.v[(outcome, l, node), s] >= sum(
                                                v_input[s][i] * coeffs_layer[node, i] for i in nodes_input) +
                                                            intercepts_layer[node]
                                                       )
                                            )
                        model.add_component('constraint_2_' + str(l) + '_' + str(node) + outcome + '_' + str(s),
                                            Constraint(rule=model.v[(outcome, l, node), s] <= M_u * (
                                                model.v_ind[(outcome, l, node), s])))
                        model.add_component('constraint_3_' + str(l) + '_' + str(node) + outcome + '_' + str(s),
                                            Constraint(rule=model.v[(outcome, l, node), s] <= sum(
                                                v_input[s][i] * coeffs_layer[node, i] for i in nodes_input) +
                                                            intercepts_layer[node] - M_l * (
                                                                    1 - model.v_ind[(outcome, l, node), s])
                                                       )
                                            )
                    v_pos_list.append(v_pos_list_s)

                ## Prepare nodes_input for next layer
                nodes_input = nodes
                v_input = v_pos_list

        if not pd.isna(ub):
            for s in range(len(S)):
                model.add_component('ub_' + outcome + '_' + str(s), Constraint(expr=model.y[(outcome, s)] <= ub))
        if not pd.isna(lb):
            for s in range(len(S)):
                model.add_component('lb_' + outcome + '_' + str(s), Constraint(expr=model.y[(outcome, s)] >= lb))

    return model


def adv_problem(model, outcome, clf_type, mfile, F, x_, lb, ub, rho, unc_type):
    """
        This function defines and solves the adversarial problem.
    """
    model.w = Var(Any, dense=False, domain=NonNegativeReals,
                  bounds=(0, 10))  # What is a better ub?
    model.mu = Var(Any, dense=False, domain=NonNegativeReals)

    if clf_type in ['rf', 'gbm']:
        model.mu_e = Var(Any, dense=False, domain=NonNegativeReals)

    def obj(model):
        if clf_type == 'mlp':
            if not pd.isna(lb):
                lam = 1
            else:
                lam = -1
            return lam * model.y['prediction']
        elif clf_type == 'cart':
            return model.mu[(outcome)]
        elif clf_type in ['rf', 'gbm']:
            # return sum([model.mu[f"{outcome}_{i}"] for i in np.unique(mfile['Tree_id'])])
            return model.mu_e[outcome]
        else:
            print('Not implemented yet')

    model.OBJ = Objective(rule=obj, sense=maximize)

    if clf_type == 'cart':
        model = constraints_tree(model, 'prediction', mfile, F, rho, unc_type, lb, ub, adversarial_algorithm_bool=True,
                                 S=[], adv_prob=True)
    elif clf_type == 'mlp':
        model = constraints_mlp(model, 'prediction', mfile, F, lb, ub, adv_prob=True, S=[])
    elif clf_type == 'rf':
        model = constraints_rf(model, 'prediction', mfile, F, rho, unc_type, lb, ub, adversarial_algorithm_bool=True,
                               S=[], adv_prob=True)
    elif clf_type == 'gbm':
        model = constraints_gbm(model, 'prediction', mfile, F, rho, unc_type, lb, ub, adversarial_algorithm_bool=True, S=[], adv_prob=True)
    if unc_type == 'l2':
        def aux_uncert(model):
            return sum((x_[i].item() - model.x[i]) ** 2 for i in F) <= rho ** 2

        model.add_component(outcome + '_aux_uncert', Constraint(rule=aux_uncert))

    elif unc_type == 'linf':
        def aux_unc_1(model, i):
            return model.x[i] <= x_[i].item() + rho

        model.add_component(outcome + '_aux_unc_1', Constraint(F, rule=aux_unc_1))

        def aux_unc_2(model, i):
            return model.x[i] >= x_[i].item() - rho

        model.add_component(outcome + 'aux_unc_2', Constraint(F, rule=aux_unc_2))

    print('Optimizing the adversarial problem...')

    opt = SolverFactory('gurobi_persistent')
    opt.set_instance(model)
    opt.set_gurobi_param('PoolSolutions', 10)
    opt.set_gurobi_param('PoolSearchMode', 1)

    # opt.options["NonConvex"] = 2
    # opt.options["DualReductions"] = 0 #########

    start_time_pp = time.time()
    results = opt.solve(model, load_solutions=True, tee=False)
    subopt_solutions = []
    print("Status:", results.solver.termination_condition)
    if (results.solver.status == SolverStatus.ok) and (
            results.solver.termination_condition == TerminationCondition.optimal):
        solution = [value(model.x[i]) for i in F]
        print('solution adv problem', solution, 'generated in ', np.round(time.time() - start_time_pp, 1), 's')

        number_of_solutions = opt.get_model_attr('SolCount')
        for i in range(number_of_solutions):
            opt.set_gurobi_param('SolutionNumber', i)
            suboptimal_solutions = opt.get_model_attr('Xn')

            vars_name_x = [opt.get_var_attr(model.x[i], 'VarName') for i in F]
            vars_name_ix = [int(vars_name_x[i].replace('x', '')) for i in range(len(vars_name_x))]
            vars_val_x = [suboptimal_solutions[i - 1] for i in vars_name_ix]
            solution_i = [vars_val_x[i] for i in range(len(vars_val_x))]
            subopt_solutions.append(solution_i)
        status = True
    else:
        solution = []
        status = False

    return model, solution, subopt_solutions, status


def master_problem(model, u, outcome, clf_type, mfile, F, rho, unc_type, lb, ub, S, time_limit = 1000):
    """
        This function defines and solves the master problem.
    """
    model_temp = add_objective(model, u, 'l1')
    if clf_type == 'cart':
        master_model = constraints_tree(model_temp, outcome, mfile, F, rho, unc_type, lb, ub,
                                        adversarial_algorithm_bool=True, S=S, adv_prob=False)
    elif clf_type == 'mlp':
        master_model = constraints_mlp(model_temp, outcome, mfile, F, lb, ub, adv_prob=False, S=S)
    elif clf_type == 'rf':
        master_model = constraints_rf(model_temp, outcome, mfile, F, rho, unc_type, lb, ub,
                                      adversarial_algorithm_bool=True, S=S, adv_prob=False)
    elif clf_type == 'gbm':
        master_model = constraints_gbm(model_temp, outcome, mfile, F, rho, unc_type, lb, ub,
                                      adversarial_algorithm_bool=True, S=S, adv_prob=False)

    print('Optimizing the master problem...')
    start_time_master = time.time()
    opt = SolverFactory('gurobi')
    try: opt.solve(master_model, timelimit=time_limit)
    except: 
        return None, None
#     opt.solve(master_model)
    end_time_master = time.time()
    sol_master = pd.DataFrame([[value(master_model.x[i]) for i in F]], columns=F)
    print('solution master', [value(master_model.x[i]) for i in F], 'generated in ',
          np.round(end_time_master - start_time_master, 1), 's')
    # print([value(master_model.y[(outcome, s)]) for s in range(len(S))])
    return master_model, sol_master


def adversarial_algorithm(model, outcome, clf_type, save_path, u, mfile, F, rho, unc_type, lb, ub, time_limit):
    """
        Adversarial algorithm with the iteration between the master problem and the adversarial problem
    """
    print('time limit: %i' % time_limit)
    if not pd.isna(ub):
        ub_price, lb_price = None, ub
        pred_ = 1
    else:
        ub_price, lb_price = lb, None
        pred_ = 0

    S = [{j: 0 for j in F}]
    iterations = 0
    # eps = 0.00001
    eps = 0
    iteration_condition = True
    comp_time = time.time()
    time_find_maxrad_list = []
    solutions_master_dict = {}
    while True:
        if time.time() - comp_time > time_limit:
            break
        print(f'\n\n------------------------ Iteration: {iterations} ------------------------')
        
#         try: master_model, sol_master = master_problem(model.clone(), u, outcome, clf_type, mfile, F, rho, unc_type, lb,
#                                                        ub, S, time_limit = time_limit)
#         except: 
#             break
        master_model, sol_master = master_problem(model.clone(), u, outcome, clf_type, mfile, F, rho, unc_type, lb,
                                                       ub, S, time_limit = time_limit)
         
        if master_model is None: 
            print('master_model is None -- MP not solved within time limit')
            break
            
        print('--> Distance to the factual instance:', value(master_model.OBJ))
        solutions_master_dict[iterations] = {'sol': sol_master, 'obj': value(master_model.OBJ)}
        time_find_max_start = time.time()
        _, dist_border = find_maxrad(sol_master, clf_type, save_path, F, [], [], {}, [], [], [],
                                                  pred_, unc_type)
        # time_find_maxrad_list.append(time.time() - time_find_max_start)
        comp_time -= time.time() - time_find_max_start
        print('--> Distance to the border:', dist_border)
        if dist_border + rho/100 >= rho:
            print("Stopping because the distance to the border is >= rho")
            break

        adv_model, sol_adv, set_sol_adv, status_adv = adv_problem(model.clone(), outcome, clf_type, mfile, F,
                                                                      sol_master, lb_price, ub_price,
                                                                      rho, unc_type)
        print('Status adversarial problem:', status_adv)
        if status_adv:
            if value(adv_model.OBJ) <= eps:
                print(f'Stopping because the ADV obj value ({value(adv_model.OBJ)}) is < eps ({eps})')
                break

            for temp_sol_adv in set_sol_adv:
                temp_sol_adv = pd.DataFrame([temp_sol_adv], columns=F)
                S_df = np.subtract(temp_sol_adv, sol_master)
                if {j: S_df[j].item() for j in F} not in S:
                    S.append({j: S_df[j].item() for j in F})
                    break
                else:
                    pass
        else:
            print('Stopping because ADV problem is infeasible')
            break
        iterations += 1

    print(f'### Iterative approach completed in {np.round(time.time() - comp_time, 1)} s ###\n')
    comp_time_final = time.time() - comp_time - sum(time_find_maxrad_list)
    model_temp = model.clone()
    if clf_type == 'cart':
        model = constraints_tree(model_temp, outcome, mfile, F, rho, unc_type, lb, ub, True, S=S, adv_prob=False)
    elif clf_type == 'rf':
        model = constraints_rf(model_temp, outcome, mfile, F, rho, unc_type, lb, ub, True, S, False)
    elif clf_type == 'mlp':
        model = constraints_mlp(model_temp, outcome, mfile, F, lb, ub, adv_prob=False, S=S)
    return model, iterations, comp_time_final, sol_master, solutions_master_dict


def add_objective(model, u, obj_type, maxrad=False):
    F = list(u.columns)

    if obj_type == 'l2':
        def l2norm(model):
            return sum((u[i].item() - model.x[i]) ** 2 for i in F)

        model.OBJ = Objective(rule=l2norm, sense=minimize)

    elif obj_type == 'linf':
        model.g = Var(['aux_g'], domain=NonNegativeReals, name=['aux_g_%s' % str(i) for i in F])

        def obj_aux1(model, i):
            return model.g['aux_g'] >= (u[i].item() - model.x[i])

        model.add_component('obj_aux1', Constraint(F, rule=obj_aux1))

        def obj_aux2(model, i):
            return model.g['aux_g'] >= -(u[i].item() - model.x[i])

        model.add_component('obj_aux2', Constraint(F, rule=obj_aux2))

        def obj_aux3(model):
            return model.g['aux_g']

        model.OBJ = Objective(rule=obj_aux3, sense=minimize)

    elif obj_type == 'l1':
        model.g = Var(F, domain=NonNegativeReals, name=['aux_g_%s' % str(i) for i in F])

        def obj_aux1(model, i):
            return model.g[i] >= (u[i].item() - model.x[i])

        model.add_component('obj_aux1', Constraint(F, rule=obj_aux1))

        def obj_aux2(model, i):
            return model.g[i] >= -(u[i].item() - model.x[i])

        model.add_component('obj_aux2', Constraint(F, rule=obj_aux2))

        def obj_aux3(model):
            return sum(model.g[i] for i in F)

        model.OBJ = Objective(rule=obj_aux3, sense=minimize)


    return model


def add_validity_constraint(model, clf, clf_type, save_path, u, F, rho, unc_type, adversarial, time_limit):
    """
        The validity constraints require an adversarial approach in the case of MLP and optionally
        in the case of tree based models.
    """
    model.y = Var(Any, dense=False, domain=Reals)
    model.l = Var(Any, dense=False, domain=Binary)
    model.y_viol = Var(Any, dense=False, domain=Binary)
    model.v = Var(Any, dense=False, domain=NonNegativeReals)
    model.v_ind = Var(Any, dense=False, domain=Binary)
    model.lam = Var(Any, dense=False, domain=Reals, bounds=(0, 1))

    clf_path = save_path + '/' + clf_type + '/model.csv'
    mfile = pd.read_csv(clf_path)

    num_iterations, comp_time = 0, 0
    lb, ub = None, None
    if clf_type in ['cart', 'gbm', 'rf']:
        if clf.predict(u) == 1:
            ub, lb = 0.4999, None
        else:
            ub, lb = None, 0.5001
    elif clf_type in ['linear', 'svm', 'mlp']:
        if clf.predict(u) == 1:
            ub, lb = 0.0, None
        else:
            ub, lb = None, 0.0

    solutions_master_dict = {}
    if clf_type in ['linear', 'svm']:
        print('embedding validity constraints...')
        model = constraints_linear(model, 'prediction', clf.predict(u), mfile, F, rho, unc_type, lb, ub)
        model = add_objective(model, u, unc_type)
        print('validity constraint embedded.')
        opt = SolverFactory('gurobi')
        start_opt = time.time()
        opt.solve(model)
        results = opt.solve(model)
        end_opt = time.time()
        solution_ = [value(model.x[i]) for i in u.columns]
        solution_ = pd.DataFrame([solution_], columns=u.columns)
        comp_time = end_opt - start_opt
    elif clf_type == 'cart':
        if not adversarial:
            print('embedding validity constraints Cart.')
            model = constraints_tree(model, 'prediction', mfile, F, rho, unc_type, lb, ub,
                                     adversarial_algorithm_bool=False, S=[], adv_prob=False)
            model = add_objective(model, u, unc_type)
            print('validity constraint embedded.')
            opt = SolverFactory('gurobi')
            start_opt = time.time()
            opt.solve(model)
            results = opt.solve(model)
            end_opt = time.time()
            solution_ = [value(model.x[i]) for i in u.columns]
            solution_ = pd.DataFrame([solution_], columns=u.columns)
            comp_time = end_opt - start_opt
        else:
            print('\n### Starting the Cart iterative approach ###')
            model, num_iterations, comp_time, solution_, solutions_master_dict = adversarial_algorithm(model, 'prediction', clf_type, save_path, u, mfile, F, rho, unc_type, lb, ub, time_limit)
    elif clf_type == 'rf':
        if not adversarial:
            print('embedding validity constraints...')
            model = constraints_rf(model, 'prediction', mfile, F, rho, unc_type, lb, ub,
                                   adversarial_algorithm_bool=False, S=[], adv_prob=False)
            model = add_objective(model, u, unc_type)
            opt = SolverFactory('gurobi')
            start_opt = time.time()
            opt.solve(model)
            results = opt.solve(model)
            end_opt = time.time()
            solution_ = [value(model.x[i]) for i in u.columns]
            solution_ = pd.DataFrame([solution_], columns=u.columns)
            comp_time = end_opt - start_opt
            print('validity constraint embedded.')
        else:
            print('\n### Starting the RANDOM FOREST iterative approach ###')
            model, num_iterations, comp_time, solution_, solutions_master_dict = adversarial_algorithm(model, 'prediction', clf_type, save_path, u, mfile, F, rho, unc_type, lb, ub, time_limit)
    elif clf_type == 'gbm':
        if not adversarial:
            print('embedding validity constraints...')
            model = constraints_gbm(model, 'prediction', mfile, F, rho, unc_type, lb, ub,
                                   adversarial_algorithm_bool=False, S=[], adv_prob=False)
            model = add_objective(model, u, unc_type)
            opt = SolverFactory('gurobi')
            start_opt = time.time()
            opt.solve(model)
            results = opt.solve(model)
            end_opt = time.time()
            solution_ = [value(model.x[i]) for i in u.columns]
            solution_ = pd.DataFrame([solution_], columns=u.columns)
            comp_time = end_opt - start_opt
            print('validity constraint embedded.')
        else:
            print('\n### Starting the GRADIENT BOOSTING iterative approach ###')
            model, num_iterations, comp_time, solution_, solutions_master_dict = adversarial_algorithm(model, 'prediction', clf_type, save_path, u, mfile,
                                                                                F, rho, unc_type, lb, ub, time_limit)
    elif clf_type == 'mlp':
        print('\n### Starting the NN iterative approach ###')
        model, num_iterations, comp_time, solution_, solutions_master_dict = adversarial_algorithm(model, 'prediction', clf_type, save_path, u, mfile, F, rho, unc_type, lb, ub, time_limit)

    return model, num_iterations, comp_time, solution_, solutions_master_dict


def generate(clf, X_train, y_train, save_path, clf_type, task_type, u, F, F_b, F_int, F_coh, I, L, P, rho,unc_type='ball', iterative=False, time_limit = 100):
    """
        This function generate the robust counterfactual explanation
    """
    write_tables(save_path, X_train, y_train, clf, clf_type, task_type)
    model = base_model(u, F, F_b, F_int, F_coh, I, L, P)
    model, num_iterations, comp_time, solution_, solutions_master_dict = add_validity_constraint(model, clf, clf_type, save_path, u, F, rho, unc_type, iterative, time_limit)

    return model, num_iterations, comp_time, solution_, solutions_master_dict
