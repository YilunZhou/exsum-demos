
# ExSum Demos

This repository contains the two rule unions that are used as examples in the [ExSum paper](https://arxiv.org/pdf/2205.00130.pdf). For more details, please consult the [documentation](https://yilunzhou.github.io/exsum/documentation.html) for the [`exsum`](https://github.com/YilunZhou/ExSum) package. 

To use all functionalities of this package, first install the required dependencies and download the `spacy` pre-trained model, via
```
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## Files
Files for SST and QQP domains closely mirror each other. 

* `*_rule_union.py` contain the `exsum.Model` object named `model`, and can be passed as the argument into `exsum` for visualization. 
* `*_rule_list.py` define all the rules in the rule unions. 
* `*_utils.py` provide helper functions for writing rules and loading data. 
* `*_explanation.pkl` are the two data files that contain the explanations as the `exsum.SetenceGroupedFEU` and `exsum.Measure` objects, ready for the construction of the `exsum.Data` object. 
* `*_explanation_raw.pkl` are the raw data instances and their feature attribution explanation values. Their data structures should be very easy to understand via command line exploration. 
* `*_generate_feu.py` demonstrate how to construct `*_explanation.pkl` from the respective `*_explanation_raw.pkl` source. 
* `*_features.py` implement feature computation used during FEU generation. 

`qqp_generate_feu.py` takes a relatively long time to finish, since there are over 40,000 instances in the QQP test set.

## New Models and Datasets

At a bare minimum, a new file (e.g. `rule_union.py`) containing the `exsum.Model` object is needed. However, its construction requires multiple steps. We recommend follow a similar process as used by the demos, of first calculating the raw explanation data, then converting them to the `exsum`-format along with feature computation, and finally loading them to the `exsum.Model`. 

## Contact

For any questions, please contact Yilun Zhou at yilun@mit.edu. 
