The current directory is a working progress which creates autograd (pytorch) manually, 
which is in BackPropExample + MicroGradMLPExample 
The latter differes from the first in that it chooses to do relu or not depending on a flag 
for reasons i will find out soon enough.

the way to read through the code is as follows 
BackPropExample -> MicroGradMLPExample -> PytorchExample -> PythorchLearningAndInfer

HigherOrderExample contains a pairwise combination which is different from the MLP approach 

PytorchExample.ipynb is an example using the pytoch library. 

since this is basically a practice to better understand a youtube series i am watching. I won't go
too indepth. I will try to understand the demo in the parent directory MICROGRAD. 

I will also try to implement micrograd and pytorch, in other settings. 

some questions worth discovering, 

how does the layer design effect training based on different questions?
do these recognize patterns? lets try mapping different operators into the input 
example: (2,4,6,8,0.5,16)
which has pattern input 1, input2, input 1 + input2, input 1 * input2, input1**input2

