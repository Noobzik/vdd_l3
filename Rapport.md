 
# VDD TP #

| Nom / Prenom   | Numéro Etudiant |
| -------------- | --------------- |
| Sheikh Rakib   | 11502605        |
| Daudin Louise  | 11606555        |

# TP1 Python VDD #

Sur le terminal taper les commandes suivant pour installer scikit

```bash
pip install -U scikit-learn scipy matplotlib
-- Pour python 3
pip3 install -U scikit-learn scipy matplotlib
```

Ensuite faire dans le terminal python :

```python
from sklearn import *
import numpy
import matplotlib.pyplot

iris = datasets.load_iris()

len(iris.data)

print(iris.target)
print(iris.feature_names)
print(iris.target_names)
```

Print the number of data, name of variables, name of classes :

```python
len(iris.data)
print(iris.feature_names)
print(iris.target_names)
```


# TP-1 version fr #

Importez les librairies numpy et preprocessing

```python
import numpy
import sklearn.preprocessing
```

Creer la matrice X suivante :

Référence : https://www.programiz.com/python-programming/matrix

```python
X = numpy.matrix('1 -1 2; 2 0 0; 0, 1, -1')
X.mean()
X.var()
```

```bash
0.4444444444444444
1.1358024691358024
```

```python
x_scaled = preprocessing.scale(X)
print(x_scaled)
```

```bash
[[ 0.         -1.22474487  1.33630621]
 [ 1.22474487  0.         -0.26726124]
 [-1.22474487  1.22474487 -1.06904497]]
```

On constate que la fonction scale ajoute des valeurs décimale au hasard.

4.   Calcule de la moyenne et de la variance de la matrice X Normalisé :

```python
X_scaled.mean()
X_scaled.var()
```
```bash
4.9343245538895844e-17
1.0
```

Lorque on a normalisé la matrice, ca permet de remettre la variance à 1.

## C Normalisation et reduction de dimensions ##

1.   Créez la matrice de données X2 suivante :

```python
X2 = numpy.matrix('1 -1 2; 2 0 0; 0 1 -1)
```

2.   Visualisez la matrice et calculez la moyenne sur les variables

```python
print(X2)
X2.mean()
```

```bash
print(X2)
[[ 1 -1  2]
 [ 2  0  0]
 [ 0  1 -1]]

X2.mean()
0.4444444444444444
```

3. Normalisez les données dans l'intervalle [0,1], Visualiser les données normalisées et calcule de moyenne sur les variables.

```python
min_max_scaler = preprocessing.MinMaxScaler()
X2_min_max = min_max_scaler.fit_transform(X2)
```

```
X2_min_max
array([[0.5       , 0.        , 1.        ],
       [1.        , 0.5       , 0.33333333],
       [0.        , 1.        , 0.        ]])
```

4.   Charger les données IRIS

```python
iris = datasets.load_iris()
```

5.   Afficher les données, les noms des variables et le nom des classes

```python
len(iris.data)

print(iris.target)
print(iris.feature_names)
print(iris.target_names)

```

6.   Visualisez les nuages de points en 2D avec des couleurs corrspondant aux classes en utilisant toutes les combinaisons de variables

```python
plt.figure()
plt.scatter(iris.data[:,0],iris.data[:,1], c = iris.target)
plt.scatter(iris.data[:,0],iris.data[:,2], c = iris.target)
```
Il y a 6 manière de faire le plot (0-1 | 0-2 | 0-3 | 1-2 | 1-3 | 2-3).

8. Analysez le manuel d’aide pour ces deux fonctions (pcaet lda) et appliquez les sur la base  Iris.  Il  faudra  utiliser pca.fit(Iris).transform(Iris)et  sauvegardez  les  résultats dans IrisPCA pour la PCA et IrisLDA pour la LDA

```python
from sklearn.decomposition import PCA
from sklearn.lda import LDA
import pandas as pd


pca_iris = PCA(n_components =2)
principalCompo_iris = pca_iris.fit_transform(iris.data)
principal_iris_df = pd.DataFrame(data = principalCompo_iris, columns = ['PComponent 1','PC2'])
#ça marche pas D:
```
 
# TP 2 #

# Exercice 1 #

```python
from sklearn.datasets.mldata import fetch_mldata

mnist = fetch_mldata('MNIST Original', data_home = custom_data_home)
```


*   figure 1

![figure1](Figures/Figure_1.png)

On a ici une représentation sous forme de matrice 20x20 des chiffres choisie aléatoirement dans une base de dimension 64.

*   figure 2

![figure2](Figures/Figure_2.png)

Nous avons ici une projection des points aléatoires des nombres.

*   Figure 3

![figure3](Figures/Figure_3.png)

Nous avons ici un ensemble principal des projections des nombres. On remarque que les points sont un peu près réunit par bloc mais pas totalement.

*   Figure 4

![figure4](Figures/Figure_4.png)

On voit déjà qu'il y a une meilleure représentation des points des nombres, en effet, il sont regroupé par une projection linéaire discriminatoire.

*   Figure 5

![figure5](Figures/Figure_5.png)

La projection isomap à réussi à regouper les points de nombres mais ce n'est pas la meilleur projection puisqu'il y a des petites quantités des nombres qui sont mélangées avec d'autres.


*   Figure 6 (Intégration linéaire des nombres locaux)

![figure6](Figures/Figure_6.png)

On a ici une intégration linéaire, mais


![figure7](Figures/Figure_7.png)


![figure8](Figures/Figure_8.png)


![figure9](Figures/Figure_9.png)


![figure10](Figures/Figure_10.png)


![figure11](Figures/Figure_11.png)


![figure12](Figures/Figure_12.png)


![figure13](Figures/Figure_13.png)


![figure14](Figures/Figure_14.png)

# Partie 2
# Exercice 1

1. Download	the	Knime	software	with	all	the	pacakges	v3.0.0
2. Load	the	Iris dataset	using	the	‘File	Reader’	node in	a	knime project	and	tests
use	several	nodes	as	Statistics	and	Plot	to	make	a	small	analysis	of	the	dataset.
* histogramme du noeud "Statistics"

![figure 1](Figures/histo.png)

* Nuage de points

![figure 2](Figures/scatterplot.png)

3. Compute	 the	 Correlation	 using	 the	 ‘Linear	 Correlation’	 node	 and	 reduce	 the
dimension	 by	 using	 the	 ‘Correlation	 Filter’.	 Plot	 the	 data	 using	 the	 selected
variables.

AF

4. Use	 the	 PCA	 and	 MDS dimensional	 reduction	 techniques	 to	 reduce	 the
dimension	 of	 the	 data	 to	 2	 variables	 and	 plot	 the	 results.	 Make	 an	 analysis
between	the	result	in	question	3	and those	obtained	here.

* Voici le resultat pour le PCA

![figure 3](Figures/pca.PNG)

* Voici le resultat pour le MDS

![figure 4](Figures/mds.PNG)

On peut constater que la méthode MDS regroupe mieux certains point (notamment ceux proche de l'Axe X) par rapport a la méthode PCA.

# Exercice 2

* **Reading full small data set** : Read large data set from DB contained nodes

* **Target Selection** : Selection: Allow to rename some columns and then select or them.

* **Baseline Evaluation** : Choose the best algorithm between three
classification algorithms: MLP, decision tree and Naïve Bayes.

* **Reduction based on LDA** : Reduces the number of columns in the input data by linear discriminant analysis

* **Auto Encoder based Reduction** :

* **Reduction based on t-SNE** : Create a probability distribution capturing the relationships between points in the high dimensional space and find a low dimensional space that resembles the probability dimension as well as possible

* **Reduction based on High Corr.** :  Identifies pairs of columns with a
high correlation (i.e. greater than a given threshold), and removes one of the two columns for each
identified pair.

* **Tree Ensemble based Reduction** :  Generate a large and carefully constructed set of trees against a target attribute and then use each attribute’s
usage statistics to find the most informative subset of features.

* **Reduction based on PCA** : Reduce all values to those which are the most
accurate for the PCA method.

* **Row Sampling** :

* **Column Splitter** : This node splits the columns of the input table into two
output tables.

* **Column Selection by Missing Values** : Remove columns with excessive
values.

* **Column Appender** : Takes two tables and quickly combines them by appending the columns of the second table to the first table.

* **Backward Feature Elimination** : Elimine une par une les valeurs les moins
pertinente en les comparants 2 par 2.

* **Forward Feature Selection** : Elimine tout et récupère une par une les
valeurs pertinente en les comparants 2 par 2.

* **Joiner** : This node joins two tables in a database-like way.

* **ROC Curve** : This node draws ROC curves for two-class classification
problems.

* **Positive class probabilities** : Select the value from the class column that stands for the "positive" class

* **Accuracies** :

* **Bar Chart** :  Visualizes one or more aggregated metrics for different data partitions with rectangular bars where the heights are proportional to the metric values.

Il y a 233 colonnes et 50 000 lignes

Du a manque de ressource nous n'avons pas pu executer le workflow, il nous est impossible d'analyser les résultats.
 
# TP 3 Classification non supervisée avec l’approche k-means #

## Exercice 1 ##

*   Ouvrez le module «k-means» sur Knime et étudiez toutes les options proposées.

Ce nœud produit les centres de clusters pour un nombre prédéfini de clusters (pas de nombre dynamique de clusters). K-means effectue une mise en cluster rigoureuse qui attribue un vecteur de données à exactement une cluster. L'algorithme se termine lorsque les affectations de clusters ne changent plus.

L'algorithme de mise en cluster utilise la distance Euclidienne sur les attributs sélectionnés. Les données ne sont pas normalisées par le nœud

Nombre de clusters : Le nombre de clusters (centres de clusters) à créer.

Centroid initialisation :

*   First k row : Initialise les centroïdes en utilisant les premières lignes du tableau d'entrée.
*   Random initialization : initialise les centroïdes à l'aide des lignes aléatoires du tableau d'entrée.
*   La case Use static random seed permet d'avoir des résultats qui sont reproductibles.

Max number of iterations : Le nombre maximum d'itérations après lequel l'algorithme se termine s'il n'a pas trouvé de solution stable auparavant.

Enable Hilite Mapping : l'hilitage d'une ligne du cluster (2ème sortie) hilitera toutes les lignes de ce cluster dans le tableau d'entrée et le tableau de la 1ère sortie.

## Exercice 2 ##

Nous proposons un scenario suivant, avec une application du PCA et son affichage par scatterplot.

---

*   Cluster 3

![Scenario](img/Scenario1.jpg)

Nous obtenons alors le résultat suivant avec la base d'iris.

![kmeans-PCA3_iris](img/kmeans-PCA_iris3.png)

Nous obtenons alors le résultat suivant avec la base waveform

![kmeans-PCA3_waveform](img/kmeans-PCA_waveform3.png)

Nous remarquons que l'on obtient 3 clusters qui ne se chevauchent pas après le traitement algorithme du cluster k-means.

---

*   Cluster 4

Nous obtenons alors le résultat suivant avec la base d'iris.

![kmeans-PCA3_iris](img/kmeans-PCA_iris4.png)

Nous obtenons alors le résultat suivant avec la base waveform

![kmeans-PCA3_waveform](img/kmeans-PCA_waveform4.png)

Nous remarquons que l'on obtient 4 clusters qui ne se chevauchent pas après le traitement algorithme du cluster k-means. Avec un PCA de 2 pour la représentation en scatterplot.

---

*   Cluster 5

Nous obtenons alors le résultat suivant avec la base d'iris.

![kmeans-PCA5_iris](img/kmeans-PCA_iris5.png)

Nous obtenons alors le résultat suivant avec la base waveform

![kmeans-PCA5_waveform](img/kmeans-PCA_waveform5.png)

Nous remarquons que l'on obtient 5 clusters qui ne se chevauchent pas après le traitement algorithme du cluster k-means.

En ce qui concerne pour la base de données d'iris, on est limité à 4 cluster.


## Exercice 3 ##

Visualisez les données initiale set ensuite visualisez les résultats du clustering.Déterminez les centres des clusters résultants. Projetez les barycentres de chaque cluster sur les résultats obtenus.

Pour la base de données iris, on a les données initiales suivantes.

![EX3_iris_init](img/EX3_iris_init.png)

---

Méthode clustering :

![EX3_iris_clustering](img/EX3_iris_clustering.png)

---

Valeurs du barycentre :

![EX3_iris_barycenter_value](img/EX3_iris_barycenter_value.png)

---

Représentation du barycentre

![EX3_iris_barycenter_value](img/EX3_iris_barycenter_chart.png)


Pour la base de donnée de waveform, on a les données initiales suivantes.

![EX3_iris_init](img/EX3_waveform_init.png)
---
Méthode clustering
![EX3_iris_clustering](img/EX3_waveform_clustering.png)
---

Valeur du barycentre :

![EX3_iris_barycenter_value](img/EX3_iris_waveform_value.png)
---

Représentation graphique du barycentre

![EX3_waveform_barycenter_chart.png](img/EX3_waveform_barycenter_chart.png)

## Exercice 4 ##

Voici le score du clustering :

![EX4_clustering_score.png](img/EX4_clustering_score.png)

Et voici le score du modèle avec la validation croisée :

![EX4_modele_score.png](img/EX4_modele_score.png)
 
# TP 4

# Exercice 1

Le PCA permet de faire une bijection des données sur une paire pour pouvoir les
visualiser en 2D (dimension to reduce: 2, si besoin, exclure les données non
pertinentes).

# Exercice 2

Voici ce que nous obtenons avec le premier scatter plot

![figure 1](Figures/Exo2SP.PNG)

Afin d'obtenir toutes les combinaisons de paires pour le scatter plot nous devons utiliser le noeud "Scatter Matrix" ce qui nous permet d'obtenir les graphes suivant :

![figure 2](Figures/MSP.PNG)

On remmarque que c'est la paire Col0:Col2 qui est la meilleure, car dans cette vue les valeurs sont mieux séparées que dans les autres vues.

![figure 3](Figures/BestView.PNG)

# Exercice 3

a) On utilise un **File Reader** afin de lire un data set puis on le relie a un noeud **PCA** afin de reduire le data set à deux dimensions

Afin de faciliter la visualisation, on utilise **Color Manager** relié avec **PCA** pour coloriser les données.

Enfin on utilisera **Scatter Plot** afin de générer un nuage de point et visualiser les données.

b)

![figure 4](Figures/PCA.PNG)

Nous constatons que les données sont mieux séparées par classes. Nous distinguons bien les 3 classes (bleu clair / bleu foncé / noir). La méthode PCA permet donc d’avoir chaque type d’iris mieux séparé et groupé.

c.d.e)

![figure 5](Figures/pca2d.PNG)
* Ci-dessus le nuage de point du PCA du data set **Waveform** a deux dimensions.

![figure 5](Figures/pca4d.PNG)
* Ci-dessus la matrice des nuages de point du PCA du data set **Waveform** a quatres dimensions.

![figure 6](Figures/pca4diris.PNG)
* Ci-dessus la matrice des nuages de point du PCA du data set **Iris** a quatres dimensions.

# Exercice 4

a) Voici toutes les combinaisons de nuage de point que nous pouvons obtenir a partir du data set **Waveform** :

![figure 7](Figures/wave.PNG)

Nous remarquons que les classes sont toutes melangées et par conséquent aucune analyse n'est possible.

b) (cf les figures de l'exercice précédent)

Nous constatons qu'en utilisant un PCA il est plus facile de distinguer les classes de données, c'est donc une méthode a favoriser pour la lecture de données.

# Exercice 5

a) (cf exercice 4)

b) Voici le resultat obtenu avec les clusters :

![figure 8](Figures/cluster1.PNG)

c) Voici le resultat avec PCA

![figure 9](Figures/clusterPCA.PNG)

d) Dans la méthode Cluster+PCA nous remarquons que les données sont regroupées autour d'un centre alors qu'avec la méthode PCA seule les données sont regroupées par classes

e)

![figure 10](Figures/barycentre.PNG)

Ci-dessus le nuage de point pour representer les barycentre
Nous n'avons pas réussi à combiner les deux nuages de point afin de superposer les cluster et leurs barycentre respectif.

![figure 11](Figures/bariris.PNG)

Ci-dessus les barycentre de la base Iris.
