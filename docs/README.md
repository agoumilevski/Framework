# Python Framework for DSGE models
 
## authors: Alexei Goumilevski (AGoumilevski@imf.org)
 
## What it is:
This Framework aims to help economists to ease development and run 
of DSGE models in Python environment.
 
 ## How to run:
 - Create or modify existing YAML model file in examples/models folder.
 - Open src/tests/test.py file and set *fname* to the name of this model file.
 - Run simulations by double-clicking on run batch file located under Framework folder.

## Content:
 - Sample model file (see `<examples/models/Toy/JLMP98.yaml>`)
 - Documentation (see `<docs/UserGuide.pdf>`)

## Highlights:
- Framework is written in Python language and uses only Python libraries that are available by installing Anaconda distribution in Software Center
- Framework is versatile to parse model  files written in a human readable YAML format, Sirius XML format and to parse simple IRIS and DYNARE model files.
- Prototype model files are created for non-linear and linear perfect-foresight models.
- It can be run as a batch process, in a Jupyter notebook, or in a Spyder interactive development environment (Scientific Python Development environment).
- Framework parses the model file, checks its syntax for errors, and generates Python functions source code.  It computes the Jacobian up to the third order in a symbolic form.
- Non-linear equations are solved by iterations by Newton's method.  Two algorithms are implemented: ABLR stacked matrices method and LBJ forward-backward substitution method.
- Linear models are solved with  Christopher Sim's method , Binder Pesaran's method, Anderson and More's method and two generalized Schur's method that reproduce calculations employed in Dynare and Iris.
- Non-linear models can be run with time dependents parameters.
- Framework can be used to calibrate models to find model's parameters. Calibration can be run for both linear and nonlinear models.  Framework applies Bayesian approach to maximize likelihood function that incorporates prior beliefs about parameters and goodness of fit of model to the data.
- Framework can sample model parameters by using Markov Chain Monte Carlo affine invariant ensemble sampler algorithm of Jonathan Goodman.
- Framework uses Scientific Python Sparse package for large matrices algebra.
- Following filters were implemented: Kalman (linear and non-linear models), Unscented Kalman, LRX, HP, Bandpass, Particle.  Several versions of Kalman filter and smoother algorithms were developed including diffuse and non-diffuse, multivariate and univariate filters and smoothers.
- As a result of runs Framework generates 1 and 2 dimensional plots and saves data in excel file and in Python sqlite database.