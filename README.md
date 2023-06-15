## Final Project of the Educational Program "Digital Methods in Energy"

Mapping of various facies zones according to GIS data (rus. Картирование различных фациальных зон по данным ГИС).

Use GIS data to locate and map facies zones of deposits. The process can be automated using machine learning.

**Aim:** build a machine learning model that finds clusters of wells from petrophysical exploration data.

### Project Description 

- In general, there may be 4-5 different clusters according to geological formation. Good point to start with.
- Received dataset consists of 39800 wells with alpha-ps GIS data and location (x-y coordinates).
- Data was corrupted and has many duplicates, which were removed after data cleaning.
- Alpha-ps GIS is a time series data that describes the petrophysical characteristics of the rock formation. It depends on the depth of the formation, so the initial single feature is time series data with *varying signal length* for the samples.
- Since different samples vary significantly in time series length, there are two possible approaches: (i) extract spectral modes from the samples and use them as features for the ML model; (ii) use a metric that is not sensitive to the length of the signal.
- First approach is based on *Fourier transform*: most important frequencies and their amplitudes are extracted and further used for clustering.
- Second approach is based on clustering with *Dynamic Time Warping (DTW) distance*: raw samples of different signal length are clustered with DTW distance. DTW is an algorithm for measuring the similarity between two temporal sequences that may differ in speed.
- The main metric used to validate the clustering result is a sihouette score (values from -1 to 1, bigger is better).

**Note:** The DTW algorithm implemented in the *dtaidistance* library produce DTW distances with all distance properties.

### Results

- Created ML models for clustering only with alpha-ps GIS data.
- Fourier space mode performed poorly (almost all samples collapsed into one cluster).
- Clustering with DTW distance produces valid and robust clustering.

### Files

- `clust_spectrum_feat.ipynb` -- workflow of clustering with Fourier space modes as features;
- `clust_timeseris_feat.ipynb` -- workflow of clustering with DTW distance;
- `clustering_sirius_14.04.pdf` -- final slides (in Russian);
- [link](https://drive.google.com/file/d/1VM8lZRWaIeMDbf4bL1yJouCnAJxb0RmP/view?usp=sharing) to the data.