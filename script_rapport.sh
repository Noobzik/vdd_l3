#!/bin/bash

if [ -e ./Rapport.md ];
then
  echo Suppression du README généré
  rm ./Rapport.md
fi

echo Generation du fichier
#cat Rapport_entete.md >> Rapport.md
for i in TP*/README.md;
do
  echo " " >> Rapport.md
  cat $i >> Rapport.md
done
