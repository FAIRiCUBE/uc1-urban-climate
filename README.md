FAIRICUBE -Urban adaptation to climate change-
==============================


About this Use Case:
-------------------------------------------------------------------------------------------

Cities face a lot of challenges combatting the impacts of climate change, such as (i) mitigating the Urban Heat Island effect; (ii) providing shading and cooling through urban green spaces and trees; or (iii) adapting to changing precipitation patterns and preparing for heavy rains and associated flash flood events. Climate change also causes pressures on (urban) biodiversity by the changes in temperature and precipitation patterns (heat waves, drought, wildfires, torrential rains, flash floods) and on agricultural surfaces and the entire agricultural system as a whole, also due to the changing patterns of temperature and rainfall. Other land use activities do also have an impact or lead to an exacerbation of the risks, such as land take, sealing of surfaces or the removal of green spaces and trees/forest. Thus, climate change together with human activities exert a lot of pressure on ecosystems, one of which are cities (urban ecosystem). Therefore, cities need to put in place concepts and measures that identify and set up clear objectives and concrete actions to mitigate the impacts and adapt to the future situation. Following the management principle “If you can’t measure it, you can’t manage it”, the basis for all actions are reliable and accessible data and information of high quality. Currently, data are coming from different sources, are of different quality and often lack metadata or information on their sources and processing. Moreover, they come in different formats which makes it difficult to combine and integrate them to derive more specific and customised information. Data cubes and the integration of data therein can be a powerful tool for cities to receive the information they need. 



Research Questions:
-------------------------------------------------------------------------------------------

•		Do the currently available European data help cities in being appropriately informed about climate change and its impact on cities? 

•		Can big data (historical, real-time, and modelled forecast spatial data) and ML approaches help European cities to prepare for the impacts of climate change and take adaptive measures/make informed decisions? 

•		In how far can datacubes enable local, regional, national, and European decision-makers to achieve the goals of the European Green Deal? 
•		Does the European Green Deal data space provide the best possible means to collect, store and provide European data on climate change impacts on cities? 



Directory layout:
-------------------------------------------------------------------------------------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
