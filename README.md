# Robust Counterfactuals Explanations
RCE is a Python package for calculating robust Counterfactual explanations (CE) for data-driven classification models. CEs play a crucial role in detecting biases and improving the explainability of these models, but most known methods can only provide a single CE which may not be practical for real-world use. Our package uses algorithmic ideas from robust optimization to provide a whole region of optimal CEs, so that the user can select the most reasonable one. Our method is proven to converge for popular ML methods such as logistic regression, decision trees, random forests, and neural networks.

The full methodology is detailed in [our manuscript](https://arxiv.org/test), Finding Regions of Counterfactual Explanations via Robust Optimization. See the [slides](https://github.com/donato-maragno/robust-CE/blob/main/slides/Finding_Regions_of_Counterfactual_Explanations_via_Robust_Optimization.pdf) for a more visual explanation of our approach. 

## How to install RCE
You can install the RCE package locally by cloning the repository and running ```pip install .``` within the home directory of the repo. This will allow you to load `rce` in Python; see the example notebooks for specific usage of the functions.

## How to use RCE 
RCE can generate robust counterfactual for logistic regression, decision trees, random forests, gradient boosting machines, and nerual networks with ReLU activation functions. The predictive models must be trained using the ```sklearn``` library.

```python
import rce
# train the classifier
clf_type = 'cart'  # supported clf types: ['cart', 'linear', 'rf', 'gbm', 'mlp']
clf = DecisionTreeClassifier(max_depth=5).fit(X_train, y_train)
# define the factual instance
u = pd.DataFrame([X_test.iloc[0, :]])
# use rce to generate robust counterfactual explanations. rce_sol is the robust counterfactual explanation.
'''
  save_path: (str) path where the clf tables will be saved
  task: (str) taks of the ML model; binary or continuous (only binary is supported at the moment)
  u: (DataFrame) factual instance
  F: (list) feature 
  F_b: (list) binary features
  F_int: (list) integer features
  F_coh: (dict) categorical features (one hot encoded)
  I: (list) immutable features
  L: (list) 'larger than' features
  P: (list) positive features
  rho: (float) dimension the uncertainty set
  unc_type: (str) shape of the robust CE; supported: 'l2' (ball) or 'linf' (box)
  iterative: (bool) if true the Robust CE can overlap more leaves otherwise it will be contained fully in one leaf. It must be true for 'mlp'
'''
final_model, num_iterations, comp_time, rce_sol, solutions_master = rce.generate(clf, X_train, y_train, save_path, clf_type, task, u, F, F_b, F_int, F_coh, I, L, P, rho,unc_type=unc_type, iterative=True)
```

## Citation
Our software can be cited as:
````
  @misc{RCE,
    author = "Donato Maragno",
    title = "RCE: Robust Counterfactual Explanations",
    year = 2023,
    url = "https://github.com/donato-maragno/robust-CE/"
  }
````

## Get in touch!
Our package is under active development. We welcome any questions or suggestions. Please submit an issue on Github, or reach us at d.maragno@uva.nl.
