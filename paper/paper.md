---
title: 'tmp'
tags:
  - Deep Learning
  - Machine Learning
  - Artificial Intelligence
  - Medical imaging
  - Clinical data
  - Computational biomedicine
authors:
  - name: Alex Golts
    corresponding: true # (This is how to denote the corresponding author)
    equal-contrib: true
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Moshiko Raboh
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Yoel Shoshan
    equal-contrib: true
    affiliation: 1
  - name: Sagi Polaczek
    affiliation: 1
  - name: Itai Guez
    affiliation: 1
  - name: Liam Hazan
    affiliation: 1
  - name: Efrat Hexter
    affiliation: 1
  - name: TBD
    affiliation: 1
  - name: TBD
    affiliation: 1
  
affiliations:
 - name: IBM Research - Israel
   index: 1
date: 26 October 2022
bibliography: paper.bib

---

# Summary

Machine Learning is at the forefront of scientific progress in Healthcare and Medicine. To accelerate scientific discovery, it is important to have tools that allow progress iterations to be collaborative, reproducible, reusable and easily built upon without "reinventing the wheel".
FuseMedML, or *fuse*, is a Python framework for accelerated Machine Learning (ML) based discovery in the medical domain. It is highly flexible and designed for easy collaboration, encouraging code reuse. Flexibility is enabled by a generic data object design where data is kept in a nested (hierarchical) Python dictionary (NDict), allowing to easily deal with information from different modalities. Functional components allow to specify input and output keys, to be read from and written to the nested dictionary.  
Easy code reuse is enabled through key components implemented as standalone packages under the main *fuse* repo using the same design principles. These include *fuse.data* - a flexible data processing pipeline, *fuse.eval* - a library for evaluating ML models, *fuse.dl* - reusable Deep Learning model architecture components and loss functions, and more. 

# Statement of need
Medical related research projects span multiple modalities (e.g., imaging, clinical data, biochemical representations) and tasks (e.g., classification, segmentation, clinical conditions prediction).
Through experience with many such projects we found three key challenges:
1. Launching or implementing a new baseline model a new baseline model can take more time than it should. This is true even when very similar projects have already been done in the past by the same lab. 
2. Porting individual components across projects is often painful, resulting in researchers “reinventing the wheel” time after time.
3. Collaboration between projects across modalities as well as across domains such as imaging and molecules is very challenging.  

FuseMedML was designed with the goal of alleviating these challenged.

Before open sourcing it, we used *fuse* internally in multiple research projects [@raboh2022context], [@rabinovici2022early], [@rabinovici2022multimodal], [@jubran2021glimpse], [@tlusty2021pre], [@golts2022ensemble] and experienced significant improvement in development time, reusability and collaboration. 
We were also able to meaningfully measure our progress and statistical significance of our results with off-the-shelf *fuse.eval* components that facilitate metrics' confidence interval calculation and model comparison. These tools were enabled us to organize two challenges as part of the 2022 International Symposium on Biomedical Imaging (ISBI) [@knight], [@bright].

# Components



check ref [@Pearson:2017]



# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

TBD

# References
