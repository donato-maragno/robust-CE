# Robust Counterfactuals Explanations
RCE is a Python package for calculating robust Counterfactual explanations (CE) for data-driven classification models. CEs play a crucial role in detecting biases and improving the explainability of these models, but most known methods can only provide a single CE which may not be practical for real-world use. Our package uses algorithmic ideas from robust optimization to provide a whole region of optimal CEs, so that the user can select the most reasonable one. Our method is proven to converge for popular ML methods such as logistic regression, decision trees, random forests, and neural networks.

## How to install RCE
You can install the RCE package locally by cloning the repository and running ```pip install .``` within the home directory of the repo. This will allow you to load `rce` in Python; see the example notebooks for specific usage of the functions.

## How to use RCE 
See notebooks in the experiments section.

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
