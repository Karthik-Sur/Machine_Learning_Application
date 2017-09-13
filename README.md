I have implemented code for a Basic Decision Tree Algorithm. I obtained the Algorithm from Dr.Vivek's Lecture. 

The Basic Decision Tree Algorithm: ID3

ID3(S, Attributes, Label):
1. If all examples have same label:
	Return a single node tree with the label
2. Otherwise
Create a Root node for tree
A = attribute in Attributes that best classifies S
for each possible value v of that A can take:
Add a new tree branch corresponding to A= v
Let Sv be the subset of examples in S with A = v
if Sv is empty:
add leaf node with the common value of Label in S
Else
below this branch add the subtree ID3(Sv, Attributes – {A}, Label)
Return Root node
In the above, the inputs are S – the set of Examples; Label is the target attribute(prediction); Attributes is the set of measured attributes
