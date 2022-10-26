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

TBD

# Statement of need

TBD



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