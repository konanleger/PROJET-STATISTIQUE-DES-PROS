#-------------------------------------------------------------------------------
##############   projet statistique  2A   ######################################
#-------------------------------------------------------------------------------

# nettoyage de l'environnement 
rm(list = ls())

# Chargement des packages necessaires
library(readxl)
library(ggplot2)
library(dplyr)

# Chargement de la base de données

data <- read_excel("makeorg_sport.xlsx")

attach(data)

# stats descriptives
str(data)
summary(data)

# Visualisation
plot(`Nb de votes`)
boxplot(`Nb de votes`~ `Âge`)

print(Proposition)



par(mfrow = c(1,3))
hist(`% pour`)
hist(`% contre`)
hist(`% neutre`)
