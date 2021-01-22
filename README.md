# Deep-Learning-Cloud-Computing

Le but du projet est de développer et héberger une application d’indexation et recherche multimédia sur ressources Cloud à l'aide de Docker, Flask et python.

Pour faire fonctionnner l'application commencez par télécharger et renommé la base d'image au nom GHIM la base d'image est disponoble au lien suivant: https://drive.google.com/file/d/1_mhv_LmSe_meN-D124B4m8WfXZIasmPK/view?usp=sharing
Placez le dossier  d'image GHIM dans le repertoire "static" du dossier de projet.

Installez les librairies necessiares au fonctionnement de l'application contenu dans le fichier requirements.txt
Exécuter le code main3.py pour lancer l'application.

# Création image Docker

Vous pouvez créer l'image docker en lancant la commande: $ docker image build -t projet_deepLearning_cloudComputing:v0.0 .$

puis lancer l'exécution de l'image docker avec la commande:  $ docker run -p 8000:8000 projet_deepLearning_cloudComputing:v0.0

site web est alors disponible à l'adresse: "votre_IP:8000"
où "votre_IP" est l'adresse IP de votre pc ou machine virtuelle
