## Introduction
Imagine you work at a hospital. You are treating a patient named John Doe and you need to look up his medical records. Upon checking the database, you stumble across this:

![](./website_imgs/intro_img1.PNG){ width=40% }

Record linkage is the process of relating entities within different datasets. An example of record linkage at a university would be relating the students within a department to the students within a class. This is made trivial by the fact that the school designates a distinct ID to each student. However, without this primary key, the task would become about using the features associated with each entry to relate the entities within each dataset.

Our framework will look to represent this machine learning problem as an edge prediction problem on a heterogeneous graph. Although the record linkage problem can be used between entities within the same relation, we will focus primarily on the task of joining two distinct relations. The reason for this is that it constrains the problem statement, allowing us to focus on the methodology before seeking for a generalizable solution. Our heterogeneous graph representation will essentially be layers of graphs.

We believe that the graphical representation will improve on traditional models of record linkage, because it allows for the expression of more complicated relationships between entities that a tabular model may not able to represent. Also, a graphical model is more modular, thus allowing for development within this project to be more easily generalizable. An example of how a graphical approach improves on a tabular is that adding multiple relations to the graph becomes more natural. Similarly, feature representation within a graphical framework is more flexible, such as representing an address based on its subparts (address, city, state, zip code). This allows for matching even in cases that there are errors within the relation, such as incorrect spelling or missing data.

The graphical approach also allows us to leverage previous research within the field, such as using the techniques described in node2vec. An example could be using the idea of negative sampling to improve the size of the training data.

## Dataset Overview (contains EDA and cleaning process)
--Wesley--

## Description of our graphs and how we created it
--Shinu--

## Baseline model
--wesley--

## Description of our final model
--Shinu--

## Comparison of our results
--Wesley--

## Any improvements we could have made
--Shinu--

## Citations
insert here
