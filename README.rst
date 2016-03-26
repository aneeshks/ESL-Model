ESLModels
=========
|Author| |License| |Maintenance|


Algorithm from The Elements of Statistical Learning book implement by Python 3 code.

Until now, I finish chapter 3.
I am working on chapter 4.

The Algorithm model is placed in ``esl_model.chx.model``, ``x`` means the number of chapter, for example,  ``esl_model.ch3.model`` 

To run the code, you must install Python >= 3.5, because I use ``@`` operate instead of ``numpy.dot``


..  code-block:: python
    
    from esl_model.ch3.model import LeastSquareModel
    
    # import prostate data set
    from esl_model.datasets import ProstateDataSet

    data = ProstateDataSet()
    
    lsm = LeastSquareModel(train_x=data.train_x, train_y=data.train_y)
    lsm.pre_processing()
    lsm.train()
    
    # after pre_processing and train, you can get the beta_hat
    print(lsm.beta_hat)

    # predict
    y_hat = lsm.predict(data.test_x)
    
    # get the test result
    test_result = lsm.test(data.test_x, data.test_y)
    
    # get the mean of square error
    print(test_result.mse)

    # get standard error
    print(test_result.std_error)


You can find the source in ``esl_model.ch3.model``  

I try to make the code clean and simple so that people can understand the algorithm easily.
 
..  code-block:: python

    class LeastSquareModel(LinearModel):
        def _pre_processing_x(self, x):
            x = self.standardize(x)
            x = np.insert(x, 0, 1, axis=1)
            return x
    
        def train(self):
            x = self.train_x
            y = self.train_y
            self.beta_hat = self.math.inv(x.T @ x) @ x.T @ y
    
        def predict(self, x):
            x = self._pre_processing_x(x)
            return x @ self.beta_hat


How to
------
I also write some article describe how to write some algorithm.  

How to write Reduced Rank LDA   
    http://littlezz.github.io/how-to-write-reduced-rank-linear-discriminant-analysis-with-python.html


Install
-------

.. code:: 

    pip(3) install git+https://github.com/littlezz/ESL-Model


Reference
---------

- The Elements of Statistical Learning 2nd Edition
- http://waxworksmath.com/Authors/G_M/Hastie/WriteUp/weatherwax_epstein_hastie_solutions_manual.pdf  
- http://www.wikicoursenote.com/wiki/Stat841  
- http://www.stat.cmu.edu/~ryantibs/datamining/lectures/21-clas2-marked.pdf
- http://sites.stat.psu.edu/~jiali/course/stat597e/notes2/lda2.pdf  
- https://onlinecourses.science.psu.edu/stat857/node/83  



.. |Author| image:: https://img.shields.io/badge/Author-littlezz-blue.svg
   :target: https://github.com/littlezz
   
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://raw.githubusercontent.com/littlezz/ESL-Model/master/LICENSE.md
   
.. |Maintenance| image:: https://img.shields.io/maintenance/yes/2016.svg
   :target: https://github.com/littlezz/ESL-Model