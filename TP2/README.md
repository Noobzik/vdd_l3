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

# Exercice 2

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
