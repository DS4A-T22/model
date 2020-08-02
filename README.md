# ASPIRE Models

This repository contains a series of Jupyter notebooks and python scripts we have coded to build, train and tune the machine learning models the ASPIRE platform uses to predict whether an asthma patient is prone to be adherent or not to the biological treatment.

Four models have been built according to the kind of data they are fed with: *Clincal*, *Pharmacological*, *Humanistic*, and *External sources*. Once built and trained, the models are serialized and stored as binary files, which then are bundled into the [ASPIRE backend server](https://github.com/DS4A-T22/backend-aspire-omnivida).    
