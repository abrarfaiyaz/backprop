Basic Experimentation with Simplest Backprogation Algorithm.

RUN smoke_test.py to understand how to run faiyaz_backprop

Experiments: 
Iterations	hidden_dim	lr	accuracy
10		2		0.1	0.756766
		6		0.1	0.845172
		10		0.1	0.848954
		20		0.1	0.8438

100		2		0.1	0.850727
		6		0.1	0.8481
		10		0.1	0.8495
		20		0.1	0.844581

1000		2		0.1	0.845645
		6		0.1	0.83867
		10		0.1	0.84044
		20		0.1	0.825435


Iterations	hidden_dim	lr	accuracy
10		2		0.01	0.756766
		6		0.01	0.756766
		10		0.01	0.756766
		20		0.01	0.756766

100		2		0.01	0.756766
		6		0.01	0.83796
		10		0.01	0.851081
		20		0.01	0.851909


From the experiment, it can be understood that lower learning rate might require higher number of iterations to converge. Better accuracy can be acquired with slow lr but large iterations. Increasing the number of hidden nodes can help accuracy when number of iteration is less.
 
