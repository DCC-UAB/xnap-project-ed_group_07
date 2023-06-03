[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11122297&assignment_repo_type=AssignmentRepo)
# XNAP-TRANSLATION GROUP
## Introducció i objectius

Aquest projecte està enfocat a la creació d'un model de màquina automàtica per a la traducció de diferents llengues com seria passar de l'anglés al català o espanyol. El principal objectiu d'aquest treball es entendre el funcionament i l'estructura interna d'aquest model de màquina automètica per poder fer-ne les modificacions pertinents per arribar a obtenir un millor model, és a dir, que sigui capaç de fer bones traduccions donades unes paraules o frases.

## Dataloader
En aquets projecte tractem tant la traducció de l'anglés al català com a l'espanyol. Les dades les tenim en el mateix format, un fitxer per parella d'idiomes on cada linia té la paraula o conjunt de paraules en un idioma (del qual volem la traducció) i després en l'altre (la traducció). 

El primer problema amb el qual ens trobem és la quantitat de les dades. En català tenim X dades les quals son relativament poques. 

D'altra banda tenim l'arxiu en esanyol que té gairebé 140.000 (139.705) però no les podem agafar totes les dades de cop ja que no era viable processar-les a la GPU alhora. Vam estar buscant altres maneres de gestionar-les sense la necessitat de processar-les totes a la vegada, al principi vam fer un bucle on anava agafant-les de 30.000 en 30.000 i guardavem els pesos cada cop que acabava el bucle perquè en la següent iteració no s'inicialitzessin a 0 sinò que agafes els de la iteració anterior. Aquesta opció ens va comportar algunes dificulatats i probles i, per tant, vam optar per canviar-ho i vam acabar fent un dataloader per agafar les dades, vam poder arribar a agafar 90.000 de les 140.000 (unn 65% de les dades).
Aquesta funció es troba al codi en l'arxiu util.py i es anomenada create_data_loader().

## Arquitecura
Tractem amb models sequence to sequence (Seq2seq) que converteixen seqüències d'un domini a un altre, com seria en el nostre cas de l'anglès al català/espanyol. Son combinacions de dos RNN, un fa d'encoder i l'altre de decoder.


## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.
El nostre projecte ha agafat com a punt de partida el codi donat el qual es tractava de 3 arxius principals: el training.py, el util.py i el predictionTranslation.py.
L'arxiu principal que crea el model es el training.py 

## Training

En aquest arxiu que importa tota la informació de l'arxiu util.py, definim principalment algunes de les variables que volem que tingui el nostre model i també és on creem i entrenem el nostre model.

Aquí llegim les dades del fitxer que volem, en aquest cas serà un arxiu amb frases en español i la seva correspondecia en anglés. N'agafem les linies per separat perqué cada una d'elles serà un dada del model.
```
data_path = './spa-eng/spa.txt' #139705
lines = open(data_path).read().split('\n')
```

To run the example code:
```
python main.py
```
## Resultats
En aquesta primera imatge hem executat el prediction.Translation amb el model creat amb 2 epochs, batch size 128, latent dim 1024, optimizer adam, lSTM ....

<img width="335" alt="image" src="https://github.com/DCC-UAB/xnap-project-ed_group_07/assets/101715910/e0880a9d-2d7b-49ab-8316-f7a81889af89">



## Contributors

Alba Ballarà    1597532@uab.cat

Berta Carner    1600460@uab.cat

Blanca de Juan  1604365@uab.cat


Xarxes Neuronals i Aprenentatge Profund
Grau de Enginyeria de Dades, 
UAB, 2023
