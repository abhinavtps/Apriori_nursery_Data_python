# Apriori_nursery_Data_python(Apriori Analysis for Association Rule Generation on nursery data for 12960 tuples)

This is a Data Mining and Machine Learning algorithm called Apriori Algorithm. It takes input and generates association rules.

# Getting Started
Clone this repo and download nursery.csv file.
Now fire up Apriori_nursery.py file which is the actual code for this algorithm.

# Prerequisites
Need to have python 3.6 installed on your machine.

# Data Source
1. Title: Nursery Database

2. Relevant Information Paragraph:

   Nursery Database was derived from a hierarchical decision model
   originally developed to rank applications for nursery schools. It
   was used during several years in 1980's when there was excessive
   enrollment to these schools in Ljubljana, Slovenia, and the
   rejected applications frequently needed an objective
   explanation. The final decision depended on three subproblems:
   occupation of parents and child's nursery, family structure and
   financial standing, and social and health picture of the family.
   The model was developed within expert system shell for decision
   making DEX (M. Bohanec, V. Rajkovic: Expert system for decision
   making. Sistemica 1(1), pp. 145-157, 1990.).

   The hierarchical model ranks nursery-school applications according
   to the following concept structure:

   NURSERY            Evaluation of applications for nursery schools
   . EMPLOY           Employment of parents and child's nursery
   . . parents        Parents' occupation
   . . has_nurs       Child's nursery
   . STRUCT_FINAN     Family structure and financial standings
   . . STRUCTURE      Family structure
   . . . form         Form of the family
   . . . children     Number of children
   . . housing        Housing conditions
   . . finance        Financial standing of the family
   . SOC_HEALTH       Social and health picture of the family
   . . social         Social conditions
   . . health         Health conditions

3. Number of Instances: 12960
   (instances completely cover the attribute space)

4. Number of Attributes: 8

5. Attribute Values:

   parents        usual, pretentious, great_pret
   has_nurs       proper, less_proper, improper, critical, very_crit
   form           complete, completed, incomplete, foster
   children       1, 2, 3, more
   housing        convenient, less_conv, critical
   finance        convenient, inconv
   social         non-prob, slightly_prob, problematic
   health         recommended, priority, not_recom

6. Missing Attribute Values: none

7. Class Distribution (number of instances per class)

   class        N         N[%]
   ------------------------------
   not_recom    4320   (33.333 %)
   recommend       2   ( 0.015 %)
   very_recom    328   ( 2.531 %)
   priority     4266   (32.917 %)
   spec_prior   4044   (31.204 %)
   
 
 # Running the tests

1.The program takes data source(nursery.csv), Minimum Support and Minimum Confidence as input.
2.Minimum Support: A minimum support is applied to find all frequent itemsets in a database.
3.Minimum Confidence: A minimum confidence is applied to these frequent itemsets in order to form rules.
4.Result: The result will show the association rules in the given dataset with the given minimum support and minimum confidence if there are any. If there are no association rules in the the set with the given support and confidence conditions, try to plug in some different (if you didn't get any results, try feeding some lower values) values of them.


