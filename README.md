 
  
   
# Dota Science

<center>A machine learning approach to clustering player styles in DOTA2</center>
    


## Executive Summary


DOTA2 is a highly complex game that is played between 2 teams of 5 opposing players, where each player can play any of 117 heroes available. This game typically poses challenges for both professional and casual players in choosing a teammate with a preferred "playing style" that would suit the team's needs. This study utilizes the unsupervised machine learning KMeans Clustering to be able to classify different "styles" of play from data collected from all 317 verified tournament players from https://www.dotabuff.com/. The results of the clustering identify 3 main clusters of playing styles: 

* <b>"Hard Carries"</b>, which is characterized by a skewed hero selection towards attack or "carry" heroes and typically have high kill rates and lower assist rates as well as being the highest ranked among the three clusters, with a median ranking of 121/317. Interestingly, most of the top 10 ranked verified tournament players belong to this cluster, perhaps showing that the ranking system is biased towards those with high games and kills.
* <b>"Versatile"</b>, which is characterized by mixed hero selection with both "carry" and "support" heroes being present in the top 5 heroes used in the cluster. These players also typically show relatively high kill rates but low number of games played, perhaps showing the lack of experience which may account for their "undecided" playing style. With regard to player rankings, they are rankged in the middle with a median ranking of 169/317.
* <b>"Hard Support"</b>, which is characterized by a strong preference to play "support" characters, and this is further supported by the top 5 heroes chosen in this cluster. This cluster shows high assist rates but low kill and death rates, that point to their playing style in game as being "support" characters that help the "carry" characters during the game. This cluster is also ranked the lowest, with its median rank being 185/317.

While the clustering was able to uncover 3 types of players, the limitations of this methodology is that there are no labels added and the granularity of the data is only until the lifetime averages of the players. A further study could be done on the match-per-match level using different supervised machine learning algorithms to create a more granular and accurate picture on the exact styles of the players. 


##### To Access the file, visit Dota Science Jupyter Notebook
