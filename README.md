
FAIRICUBE Use Case "Urban adaptation to climate change"
==============================

- [FAIRICUBE Use Case "Urban adaptation to climate change"](#fairicube-use-case-urban-adaptation-to-climate-change)
  - [Getting started](#getting-started)
    - [Install `src` package](#install-src-package)
    - [Path to `s3/data`](#path-to-s3data)
  - [Directory layout](#directory-layout)
  - [About this Use Case](#about-this-use-case)
    - [Research Questions](#research-questions)
    - [Implementation methodology](#implementation-methodology)
    - [Expected results](#expected-results)

Getting started
-------------------------------------------------------------------------------------------

### Install `src` package

The `src` folder contains reusable scripts and functions. To use them in the notebooks, you must declare `src` as a package.

- Open a terminal window
- Navigate to the project's root folder
- Install `src` as editable package with the command `pip install -e .`

You can now import `src` as any other package. For example, to use `utils` functions in notebooks, `import src` and call the function `utils.<function-name>`. Edits to the `src` folder are immediately available in the notebooks.

### Path to `s3/data`

FAIRiCube UC1 team members have access to an S3 bucket mounted in their EOXHub workspace at `/s3`. Data in the S3 bucket can be accessed by specifying its relative path. For example, to access a file `test.csv` saved in the directory `s3/data` from a script in the `/processing` directory, use the following path

```Python
test_file = './../../s3/data/test.csv'
```

Directory layout
-------------------------------------------------------------------------------------------

    ├── data               <- sample data. Note: all data is stored in an S3 bucket
    │
    ├── notebooks          <- Jupyter notebooks
    │   ├── demo           <- notebooks to show how to use the software developed in this repository
    │   └── dev            <- work-in-progress notebooks for testing and developing
    │
    ├── pre-processing     <- tools for data pre-processing. Can include Python scripts, CLI instructions etc.
    │
    ├── processing         <- tools for data processing. Mainly Python scripts
    │
    ├── post-processing    <-  tools for data post-processing. Can include Python scripts, CLI instructions etc.
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── utils.py       <- Collection of common functions used in notebooks
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
    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
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
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

About this Use Case
-------------------------------------------------------------------------------------------

As reported by the most recent EEA report on urban adaptation to climate change, cities face a lot of challenges combatting the impacts of climate change, such as (i) mitigating the Urban Heat Island effect; (ii) providing shading and cooling through urban green spaces and trees; or (iii) adapting to changing precipitation patterns and preparing for heavy rains and associated flash flood events. Climate change also causes pressures on (urban) biodiversity by the changes in temperature and precipitation patterns (heat waves, drought, wildfires, torrential rains, flash floods) and on agricultural surfaces and the entire agricultural system. Other land use activities do also have an impact or lead to an exacerbation of the risks, such as land take, sealing of surfaces or the removal of green spaces and trees/forest. Thus, climate change together with human activities exert a lot of pressure on ecosystems, one of which are cities (urban ecosystem). Therefore, cities need to put in place concepts and measures that identify and set up clear objectives and concrete actions to mitigate the impacts and adapt to the future situation. Following the management principle “If you can’t measure it, you can’t manage it”, the basis for all actions are reliable and accessible data and information of high quality. Currently, data are coming from different sources, are of different quality and often lack metadata or information on their sources and processing. Moreover, they come in different formats which makes it difficult to combine and integrate them to derive more specific and customised information. Data cubes and the integration of data therein can be a powerful tool for cities to receive the information they need.

### Research Questions

- Do the currently available European data help cities in being appropriately informed about climate change and its impact on cities?
- Can big data (historical, real-time, and modelled forecast spatial data) and ML approaches help European cities to prepare for the impacts of climate change and take adaptive measures/make informed decisions?
- In how far can datacubes enable local, regional, national, and European decision-makers to achieve the goals of the European Green Deal?
- Does the European Green Deal data space provide the best possible means to collect, store and provide European data on climate change impacts on cities?

### Implementation methodology

The use case covers “cities” as spatial entities of analysis. The term “city” can be understood differently by different people. It can be an administrative unit (delimited by the administrative border), a morphological unit (characterized by the urban fabric) or a functional unit (adding the commuting zone to the core city from where people travel into the city for work). On the European level, the reference units for cities are the “City” which corresponds to the core city (an administrative unit), the “Commuting Zone” (also delimited by administrative borders but identified according to their function) and the “Functional Urban Area (FUA)” which is the sum of the two (i.e., aggregating the city and commuting zone). We will use this definition in our use case.

Basically, cities need to (i) be informed about the available data and get access to them in an easy-to-use way, (ii) receive accurate and up-to-date information about the current situation, (iii) understand the impacts of decisions they take, and (iv) learn about best practice examples when taking decisions. Many data points need to be tracked to properly react to how and why a city is changing over time. Comprehending a city’s baseline data allows city planners to make decisions that offer maximum benefits to its inhabitants. Within the use case we try to provide a tool kit to cities to make better informed decisions by having us much data as possible at their fingertips and being able to simulate their decision-making process.

The implementation follows several steps:

- Step 1: Collect European data, resample to 10m resolution and upload to the cube system (if not there already, e.g., in a DIAS with direct link to our cube system); at the same time, identify and assess local approaches to climate impact data collection and dissemination.
- Step 2: Process the European level by creating dashboards with indicators/indices describing the status quo (status and trends of parameters) of cities (factsheet), thus being comparable across Europe since based on European data; if possible, also modeled data need to be ingested (such as climate models, flood risk models, population - think of exposure of people and assets, both existing and planned, to changing flood risks).
- Step 3: Zoom in to the local/city level and collect local data and indicators based on stakeholder exchange and requests; harmonise them with other data across spatio-temporal and thematic content
- Step 4: Identify the impact of decisions taken by cities - what happens if one parameter changes intentionally by decisions, does it get better or worse? Which parameter has the strongest influence on improving the quality of life (city- and location-dependent)? If possible, integration of ML approaches (simulations/models/scenarios) into the future with which we can simulate possible outcomes and the consequences thereof.

Lastly, we will assess the applicability, usefulness and quality of the new data cube and the calculated products (e.g., indicators, visualization tools) compared to the existing information; assess the compliance of the data cube with the European Green Deal data space requirements.

As mentioned under the project objectives, the use case will be a demonstrator of the FAIRiCUBE HUB to allow for broader use and better usability of data cubes. It will enable a broader range of stakeholders to focus on what they are supposed to do best: overcome technical barriers to make data-driven decisions and leverage state-of-the-art processing technologies, including machine learning (ML). Although each use case is thematically different, all aspects of its operability from data requirements to output results and validation will be coordinated and supervised to ensure the correct synergies between thematic areas. To enable synergies between the individual use cases, three of the four use cases are collocated, focusing on the functional urban areas of Oslo, Vienna and Barcelona, with an area ranging between 5000 and 16000km². Where possible, data sources, processing and ingested data will be re-used. Validation of each use case will be performed. Satisfactory and unsatisfactory outcomes will be shared across use cases for mutual benefit. This will be achieved by strong coordination and supervision across use cases.

### Expected results

The project “Use case on Urban adaptation to climate change” will provide:

- a dedicated database/cube system containing all relevant available data (10m and 100m spatial resolution).  
- Dashboards with indicators/indices describing the status quo (status and trends of parameters)
- ML outputs, simulations or modelling of changes in the overall system when turning one screw a little bit: which parameter(s) has/have the strongest influence on improving the quality of life in cities in times of climate change?
- Factsheets, policy briefs, reports

Being a demonstrator for the FAIRiCUBE HUB, the use case will showcase how such an environment can enable stakeholders to access and analyse data utilising ML: offering users access to data, provided as data cubes, machine learning models tailored for data cubes, and specific services required by the use cases. The solution will be built upon renowned and proven technologies and services like DIAS and Euro Data Cube.
