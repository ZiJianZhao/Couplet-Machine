This is a project contains the data, neural network program using MXNet and website code for a couplet machine.

The data directory contains the original data sources, processed final data, and some python scripts for crawling websites or processing the data.

The couplet_model direcotory contains the program written in MXNet. The model is under sequence to sequence framework with focus attention mechanism. And we add POS knowledge using pyltp and rhyme information using pypinyin.

The couplet_website contains the website written in Flask.