This is python code for reproducing the proof-of-concept experiments of the paper
"Causal Regularisation" by Dominik Janzing, NeurIPS 2019. 
Since it is based on a rather simplistic toy scenario it is not clear whether it yields results of practical relevance. The goal of the NeurIPS paper was just to stimulate a discussion on using regularisation also against confounding apart from the usual way of using it against overfitting. 

 
### Prerequisites
Python 2.9


### How to run
Just call 

     run_ConCorr_on_real_or_simulated_data.py 
     
and you will be asked in the menu which of the experiments in the NeurIPS paper you want to reproduce.

    Which experiment from the NeurIPS paper you want to reproduce?
    Section 4.1, simulated data (1)
    Section 4.2, real data from the optical device (2)
    Section 4.2, real data on taste of wine (3)

- If you answer 1, one of the 4 scatter plots in Figure 2 will be generated according to which option you choose 
In the menue 
    
        ridge cv / lasso cv / ridge concorr / lasso concorr

- If you answer 2, the scatter plots in Figure 3 will be generated after answering the following two questions 12 times:
  you should answer 'n' to 'normalize data y/n' and '9' to 'which component should be dropped'
  (Since component 9 is the known confounder).
  After closing the first plot (showing the results for Ridge) you obtain the second plot with the result for Lasso.  
   
- If you answer 3, you need to answer 'y' to 'normalize data y/n' and obtain the values given in the paragraph 'Taste of wine'

There are some numerical deviations from the values obtained in the paper which may be due to some modification of the details of the regression procedure.  
 





