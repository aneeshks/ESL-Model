ESLModels
=========
|Author| |License| |Maintenance|


Algorithm from The Elements of Statistical Learning book implement by Python 3 code.

Until now, I finish chapter 3.

The Algorithm model is placed in ``esl_model.chx.model``, ``x`` means the number of chapter, for example,  ``esl_model.ch3.model`` 

To run the code, you must install python > 3.5, because I use ``@`` operate instead of ``numpy.dot``


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


Install
-------

.. code:: 

    pip install git+https://github.com/littlezz/ESL-Model


Reference
-----------
- The Elements of Statistical Learning 2nd Edition

- http://waxworksmath.com/Authors/G_M/Hastie/WriteUp/weatherwax_epstein_hastie_solutions_manual.pdf


.. |Author| image:: https://img.shields.io/badge/Author-littlezz-blue.svg
   :target: https://github.com/littlezz
   
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://raw.githubusercontent.com/littlezz/ESL-Model/master/LICENSE.md
   
.. |Maintenance| image:: https://img.shields.io/maintenance/yes/2016.svg
   :target: https://github.com/littlezz/ESL-Model