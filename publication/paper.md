---
title: "Pangeo-fish: A Python package for studying fish movement using bio-logging and earth science data"
tags:
  - Python
  - Pangeo
  - Dask
  - Xarray
  - Kerchunk
  - biologging
  - geolocation
  - Earth science
authors:
  - name: Justus Magin
    orcid: 0000-0002-4254-8002
    affiliation: 1
  - name: Tina Odaka
    orcid: 0000-0002-1500-0156
    affiliation: 1
  - name: Marine Gonse
    orcid: 0000-0002-5378-8482
    affiliation: 2
  - name: Jean-Marc Delouis
    orcid: 0000-0002-0713-1658
    affiliation: 1
  - name: Mathieu Woillez
    orcid: 0000-0002-1032-2105
    affiliation: 2
affiliations:
  - name: LOPS (Laboratory for Ocean Physics and Satellite remote sensing) UMR 6523, Univ Brest-Ifremer-CNRS-IRD, Plouzané, France
    index: 1
  - name: DECOD (Ecosystem Dynamics and Sustainability), IFREMER-Institut Agro-INRAE, Plouzané, France
    index: 2
date: 21 March 2024
bibliography: paper.bib
---

# Summary

Geo-referenced data plays an important role in understanding and conserving natural resources, particularly when investigating biological phenomena such as fish migration and it's habitats. Biologging, the practice of attaching small devices to animals for behavior tracking and environmental data collection, proves invaluable in this field. However, directly tracking fish underwater presents persistent challenges. To address this, models have emerged to estimate fish locations by correlating data from biologging devices—such as temperature and pressure readings—with ocean temperature and bathymetry models. The accuracy and resolution of these reference datasets significantly impact the precision of reconstructed fish trajectories. Despite recent advancements in earth observation technology and modeling methodologies like digital twins, accessing vast earth science datasets remains cumbersome due to their size and diversity. Additionally, the computational demands for analysis pose technical barriers. The Pangeo ecosystem was created by a community of engineers and geoscientists specifically to address these big earth data analysis challenges. Pangeo-fish is a Python package that utilizes Pangeo to leverage advancements in biologging data analysis for fish.

# Statement of need

Biologging, the process of attaching small devices to animals to monitor their behaviour and collect environmental data, is an important tool for understanding animal habitats.

However, unlike animals, which can be tracked using GPS technology, tracking fish underwater presents significant challenges. This limitation hinders the accurate delineation of protected areas, which is crucial for the protection of important fish habitats.

To address this issue, various tagging experiments have been conducted on a variety of fish species.

Archival tags and acoustic tags are two common tagging systems used in various projects. Archival tags, implanted in marine animals, record and store a wide range of data including temperature, pressure, light levels and salinity. Similarly, acoustic tags emit signals and are implanted in marine animals to provide location information when fish come within range of acoustic detection devices.
The computation of fish trajectories depends on the likelihood of observed data from fish tags, such as temperature at specific depths, alongside reference geoscience data such as satellite observations.

The use of reference data with high spatial and temporal resolution can significantly improve the accuracy of reconstructed fish tracks. However, handling such high resolution data requires significant computing power, storage capacity, parallelization of computations and improved data access patterns.

The Pangeo community is dedicated to fostering an ecosystem of interoperable, scalable, open source tools for interactive data analysis in the field of big data geoscience. Leveraging the Pangeo ecosystem provides an opportunity to address the challenges faced in biologging. Pangeo-fish utilises various Pangeo components, including a user-friendly interface such as JupyterLab, a robust data model such as Xarray, kerchunk and zarr, and a scalable computing system such as Dask.

By using libraries such as intake, kerchunk and fsspec, data loading processes are streamlined. In addition, Xarray and Dask facilitate computations, while visualisation tools such as hvplot and Jupyter enable interactive visualisation of results. The Pangeo software stack provides researchers with the necessary tools to access data and compute high-resolution fish tracks in a scalable and interactive manner, while giving biologists the flexibility to choose their preferred platform for analysis execution, data access and computing system, whether on a traditional HPC system, in the public cloud or on a laptop.

# Mathematics

Our approach follows the methods established [@woillez_hmm-based_2016]. It incorporates the use of a Hidden Markov Model (HMM) to quantify uncertainties and derive the posterior probability of the sequence of states (fish positions).

Here, $P(Y_t|X_t)$ express the observation likelihood, $P(X_t|X_{t-1})$ as the state prediction, where $t$ is the time, $X_t$ the hidden states and $Y_t$ the observations a time $t$.

The hidden staes $X_t$ corresponds to fish positions.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:

- `@author:2001` -> "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

T Odaka, JM Delouis and J Magin thanks to the support by CNES Appel a projet R&T R-S23/DU-0002-025-01.
T Odaka, JM Delouis and M Woillez thanks to the support by the TAOS project funded by the IFREMER via the AMII OCEAN 2100 programme.

# References
