1. # **Sources**
1) Article scientifique :

François Kawala, Ahlame Douzal, Eric Gaussier et Eustache Diemert, « Prédictions d’activité dans les réseaux sociaux en ligne », site internet [*https://hal.science/*](https://hal.science/), le 12 novembre 2013, [*https://hal.science/hal-00881395v1/document*](https://hal.science/hal-00881395v1/document) (01/12/2022).

2) Description des données :

« Buzz in social media Data Set », site internet [*https://archive.ics.uci.edu/*](https://archive.ics.uci.edu/) , le 27 mai 2013, [*https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media*](https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media)[+#*](https://archive.ics.uci.edu/ml/datasets/Buzz+in+social+media+#)  *(01/12/2022).*

3) Dataset choisi :

« regression.tar.gz/Twitter/Twitter.data » renommé « Twitter-data.csv » dans le notebook(4), téléchargeable depuis site internet [*https://archive.ics.uci.edu/*](https://archive.ics.uci.edu/) , le 27 mai 2013, [*https://archive.ics.uci.edu/ml/machine-learning-databases/00248/*](https://archive.ics.uci.edu/ml/machine-learning-databases/00248/) *(01/12/2022).*

4) Notebook du compte-rendu :

 Twitter-analyses.ipynb, le 13 janvier 2023*.*
2. # **Description du dataset**
Notre sujet vise à prédire les potentiels buzz sur les réseaux sociaux. A priori, il s’agit d’un problème de régression.

**Comment ont été choisies les variables de notre dataset ? (1)**

D’après l‘article (1), le choix des attributs vise plusieurs objectifs :

- Être utilisable pour différents sites dédiés aux réseaux sociaux dans différentes langues ;
- Passer à l’échelle pour anticiper la croissance des réseaux sociaux ;
- Ne pas se reposer sur une connaissance a priori du graphe des utilisateurs du réseau social cible, car cette information est rarement disponible.

**Quelles sont les sources de nos données ?**

- TomsHardware ;
- Twitter.

Pour nous heurter aux problématiques de quantité, nous choisissons de travailler sur les données issues de Twitter.

**Quels sont les attributs du dataset ?**

Contrairement à ce qui est annoncé dans les différents descriptifs (1)(2) fournis avec le dataset, le dataset contient 583 250 individus en dimension 78, dont un attribut cible.

Ici, on peut en fait considérer que les attributs du dataset sont au nombre de 11. Ils sont appelés « descripteurs ». Les descripteurs proposés dépendent du temps et nous pouvons les utiliser pour définir une série temporelle multivariée unique :

- Un individu du dataset correspond à un sujet, ou « topic », déterminé par mot clefs (probablement le #hashtag sur Twitter) ;
- Le libellé des topics n’apparaît pas dans les variables - les individus sont "anonymisés" ;
- Pour Twitter, chaque descripteur est décliné sur 7 jours consécutifs ;
- D’où, 7 jours  \* 11 descripteurs + 1 cible = 78 colonnes ;
- Les variables sont nommés par "préfixe du descripteur"\_"num jour", *ex : NAD \_ 3* ;
- L’indice des variables représente donc une donnée temporelle (en jours pour Twitter).

**Quelle est la signification de chaque variables ?**

- NCD : Number of Created Discussions  (columns [0,6])
- AI(NA) : Author Increase(AI) (columns[7, 13])
- AS(NAD) : Attention Level (measured with number of authors) (columns [14,20])
- BL(NCD, NAD) : Burstiness Level (columns [21,27]) -> binaire ?
- NAC : Number of Atomic Containers (NAC) (columns [28,34]) -> Tweet ?
- AS(NAC) : Attention Level (measured with number of contributions) (columns [35,41])
- CS : Contribution Sparseness (columns [42,48]) -> binaire
- AT(NA, NAD) : Author Interaction (columns [49,55])
- NA : Number of Authors (columns [56,62])
- ADL(NAD) : Average Discussions Length  (columns [63,69])
- NAD : Number of active discussion (columns [70,76])
- MNAD (Y) : Mean Number of active discussion  : Annotation (column77)

Remarques a priori :

- D’après l’article (1), 6 attributs, parmi les 11, semblent être des constructions mathématiques intégrant au moins un autre attribut du dataset (nous les avons représenter entre parenthèses dans la liste ci-dessus) ;
- MNAD est construit mathématiquement par une prédiction Regression Random Forests avec validation croisée répétée cinq fois sur l’ensemble des exemples.
3. # **Approche suivie**
Notre travail suit les étapes listées ci-dessous. Le script est fourni sous forme de notebook (4) :

- Charger les données Twitter-data.csv
- Étiqueter les colonnes du dataset ;
- Décrire le dataset ;
- Gérer les valeurs NaN ;
- Réduire le nombre de données à 1% du dataset total ;
- Partager le dataset en deux échantillons Train (70%)/Test(30%) ;
- Standardiser les données en prévision d’un réseau de neurone (non réalisé) ;
- Découper le dataset (suivant la série temporelle ou suivant les descripteurs) pour être manipulé de différentes manières ;
- Observer quelques relations entre descripteurs et jours avec l’outil facet grid ;
- Observer pairplot de descripteurs, pour un jour fixé ;
- Observer pairplot de jours, pour un descripteur fixé ;
- **Round 0 : Proposer un clustering** en 2 groupes (supposés « buzz » et « non-buzz ») ;
- Observer le clustering dans un pairplot ;
- Vérifier la pertinence des groupes (ARI très proche de 1) ;
- **Round 1 : Utiliser la méthode Nearest Neighbors Colored Classification** (non abouti) ;
- **Round 2 : Utiliser la méthode KNN** (score = 0,9 avec knn = 5) ;
- Optimiser le k-neighbors (score = 0,95 avec knn = 3) ;
- Tester le modèle optimisé (score = 0,88) ;
- **Round 3 : Réduire le nombre d’attributs** (+1%) ;
- **Round 4 : Utiliser la méthode CART** (score ~ 0,7 très instable, feuilles = 5) ;
- Optimiser le nombre de feuilles (score ~ 0,9 instable, feuilles = 15) ;
- Tester le modèle (score ~0,9 % instable) ;
- **Round 5 : Utiliser la méthode Decision Tree Classifyer** (non abouti).
4. # **Enseignements**
- Le dataset n’a pas de données manquantes.
- Le nombre d’individus est trop grand pour manipuler le dataset aisément durant la phase d’étude, donc nous avons choisi de le réduire à 1 % de son total, soit environ 5 800 individus. C’est sûr ce dataset réduit que nous séparerons données d’entraînement (4000 individus, soit 70%) et données de test (1800 individus, soit 30%).
- L’outil facet grid permet de se faire une idée de quelques caractéristiques de notre problème. Par exemple, un sujet qui n’a pas beaucoup d’engouement dans les premiers jours n’explosera jamais au bout d’une semaine. Ou encore, les sujets qui ont peu de contributeurs sont bien plus nombreux  que les sujets avec beaucoup de contributeurs.
- Les valeurs descriptives de chacun des attributs (max, mean, std) sont plutôt croissantes au long des 7 jours de mesures.
- Le dataset qui présente une problème de régression peut être transformé en problème de classification. En effet, la target étant une valeur numérique quantifiant la propension d’une thématique à devenir un Buzz, on peut établir une valeur seuil à partir de laquelle un individu serait qualifié de Buzz. Ce qui permet de transformer la target en label (« buzz » ou « non buzz »). La matrice de confusion donne un résultat très satisfaisant. Ce qui est confirmé par l’ARI.
- Nous avons établi ce seuil (MNAD > 4000) en nous appuyant sur la méthode de kmeans clustering.
- D’après ce clustering, environ 1 individu sur 200 serait un buzz dans notre dataset.
- Les différents pairplot indiquent que plusieurs attributs sont liés entre eux. Lorsque qu’on associe le pairplot à l’attribut label « Buzz or not », on remarque que les individus qualifiés de « buzz » montrent les plus grandes valeurs dans tous les domaines. Cette tendances ne varie pas en fonction des jours de la série.
- La méthode des k plus proches voisins montre de très bons résultats (souvent supérieurs à 90 %, en fonction du kfold). Le nombre de voisins optimal est 3. Il a été sélectionné pour la stabilité de ses performances grâce à plusieurs étapes de recherche d’optimisation. Ce paramètre optimisé permet d’améliorer les scores de prédiction de 5 % à 12 %, en fonction des données de test.
- Le dataset constitué de 77 (11 descripteurs) attributs peut être restreint à 21 attributs (3 descripteurs) sans perte de qualité. Au contraire, cela améliore les score de prédiction de 1 à 2 %.
- La méthode CART peut montrer de très bons résultats (supérieur à 90 %) mais ne présente pas de stabilité satisfaisante (souvent inférieur à 70%). Le nombre de feuille optimal est 15. Il est choisi pour la stabilité de ses performances grâce à plusieurs étapes de recherche d’optimisation. Ce paramètre optimisé permet théoriquement d’améliorer les scores de prédiction. Mais le manque de stabilité des scores rend plus difficile l’évaluation de l’amélioration du modèle (voir notes de conclusion). Néanmoins, les scores aberrants paraissent beaucoup moins fréquents et la plupart tourne autour de 80 %.
1. # **Conclusion**
Pour la stabilité et la performance apparentes de ses résultats de prédiction nous privilégierons le modèle KNN avec K = 3. Ainsi, notre modèle donnera une prédiction de ~90 %.

NB :

- Les estimations de performance de nos modèles de prédiction n’ont pas été réalisées grâce à des mesures rigoureuses sur un grand nombre de tirages. Nous pourrions approfondir ce point afin de poursuivre notre travail.
- Nous pourrions aussi tester nos modèles sur les 99 % restant du dataset (préalablement réservés pour des raisons pratiques) afin de vérifier leur pertinence sur un très grand nombre de données.
- Enfin, nous pourrions confirmer le choix des 3 « sélecteurs suffisants » sur les 11 initiaux, par une méthode plus rigoureuse (en retirant successivement les attributs et en comparant les variations sur la performance du modèle, par exemple).

