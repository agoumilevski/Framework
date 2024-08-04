# Python Framework for DSGE models
 
## authors: Alexei Goumilevski (AGoumilevski@hotmail.com)
 
## What it is:
This Framework aims to help economists to ease development and run 
of DSGE models in Python environment.
 
 ## How to run:
 - Create or modify existing YAML model file in Framework/models folder.
 - Open src/tests/test.py file and set *fname* to the name of this model file.
 - Set working firectory to Framework/src.
 - Run script tests/test.py.

## Content:
 - Sample model file (see `<models/Toy/JLMP98.yaml>`)

## Highlights:
- Framework is written in Python language and uses only Python libraries that are available by installing Anaconda distribution in Software Center
- Framework is versatile to parse model  files written in a human readable YAML format, Sirius XML format and to parse simple IRIS and DYNARE model files.
- Prototype model files are created for non-linear and linear perfect-foresight models.
- It can be run as a batch process, in a Jupyter notebook, or in a Spyder interactive development environment (Scientific Python Development environment).
- Framework parses the model file, checks its syntax for errors, and generates Python functions source code.  It computes the Jacobian up to the third order in a symbolic form.
- Non-linear equations are solved by iterations by Newton's method.  Two algorithms are implemented: ABLR stacked matrices method and LBJ forward-backward substitution method.
- Linear models are solved with  CBinder Pesaran's method and two generalized Schur's method that reproduce calculations employed in Dynare and Iris.
- Non-linear models can be run with time dependents parameters.
- Framework uses Scientific Python Sparse package for large matrices algebra.
- Following filters were implemented: Kalman (linear and non-linear models), Unscented Kalman, LRX, HP, Bandpass, Particle.  Several versions of Kalman filter and smoother algorithms were developed including diffuse and non-diffuse, multivariate and univariate filters and smoothers.
- As a result of runs Framework generates 1 and 2 dimensional plots and saves data in excel file and in Python sqlite database.

  ## DISCLAIMERS
  - THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE
