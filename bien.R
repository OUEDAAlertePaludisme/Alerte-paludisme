# 1. Redémarrer R d'abord (Ctrl+Shift+F10 dans RStudio) pour repartir
#    d'une session propre — évite les conflits d'initialisation reticulate/Python.

# 2. Se placer dans le dossier de l'application
setwd("D:/Alerte_paludisme/application_palu")

# 3. Lancer l'application
shiny::runApp("app.R")