# CAFA-5-Protein-Function-Prediction
___________________________________________________________________________________________________________________________________________________________________________________________________


Proteins are responsible for many activities in our tissues, organs, and bodies and they also play a central role in the structure and function of cells. Proteins are large molecules composed of 20 types of building-blocks known as amino acids. The human body makes tens of thousands of different proteins, and each protein is composed of dozens or hundreds of amino acids that are linked sequentially. This amino-acid sequence determines the 3D structure and conformational dynamics of the protein, and that, in turn, determines its biological function. Due to ongoing genome sequencing projects, we are inundated with large amounts of genomic sequence data from thousands of species, which informs us of the amino-acid sequence data of proteins for which these genes code. The accurate assignment of biological function to the protein is key to understanding life at the molecular level. However, assigning function to any specific protein can be made difficult due to the multiple functions many proteins have, along with their ability to interact with multiple partners. More knowledge of the functions assigned to proteins—potentially aided by data science—could lead to curing diseases and improving human and animal health and wellness in areas as varied as medicine and agriculture

here in this we have to predict the function of a set of proteins. You will develop a model trained on the amino-acid sequences of the proteins and on other data. Your work will help ​​researchers better understand the function of proteins, which is important for discovering how cells, tissues, and organs work. This may also aid in the development of new drugs and therapies for various diseases.

for this we have implemented model of CAFA 5 Protein Function Prediction with CNN TensorFLow via distributed processing and we use library called "DASK" memory-efficient computing.provide memory-efficient alternatives to Pandas for working with large datasets. They enable distributed computing or lazy evaluation, which can help mitigate memory issues.

we go two kinds of implementations one is Sequential Api network or Functional Api and another is CNN based model and we acheive better performance comparing to normal sequential api and functional api network but only one problem in CNN based models during tranation of models is memory issues for that we use library which i already mentioned in above...

and at last we tested the model with separate testing dataset rather than entire dataset separation ..

