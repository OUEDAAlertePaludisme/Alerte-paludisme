library(shiny)
library(bslib)
library(shinyWidgets)
library(leaflet)
library(sf)
library(dplyr)
library(plotly)
library(DT)
library(reticulate)
library(officer)
library(flextable)
library(blastula)
library(httr)
library(jsonlite)
library(readxl)
library(waiter)
library(emayili)

source("R/mod_dashboard.R")
source("R/mod_prediction.R")   # ← remplacer par mod_prediction_v2.R renommé
source("R/mod_carte.R")
source("R/mod_future.R")       # ← remplacer par mod_future_v2.R renommé
source("R/mod_alertes.R")
source("R/mod_rapport.R")

reticulate::use_python("C:/Users/OUEDA/anaconda3/python.exe", required = FALSE)

DATA_PATH      <- "data/sant.xlsx"
SHAPEFILE_PATH <- "shapefiles"
PIPELINE_PATH  <- "python"

# ── Helpers de conversion reticulate → data.frame ────────────────────────────
conv_df <- function(raw_list) {
  as.data.frame(do.call(rbind,
    lapply(raw_list, function(x) as.data.frame(x, stringsAsFactors = FALSE))
  ))
}

appliquer_strategie_future <- function(py, strat, seuils_df) {
  # Appliquer la stratégie sur future_pred_raw (liste Python brute)
  # en passant par apply_strategy_future côté Python
  tryCatch({
    fp_raw     <- py$predict_future(6L)
    fp_updated <- py$apply_strategy_future(fp_raw, strat)
    fp         <- conv_df(fp_updated)
    fp$annee         <- as.integer(unlist(fp$annee))
    fp$mois          <- as.integer(unlist(fp$mois))
    fp$Pred_ML       <- as.numeric(unlist(fp$Pred_ML))
    fp$Niveau_Alerte <- as.character(unlist(fp$Niveau_Alerte))
    fp$Nom_DS        <- as.character(unlist(fp$Nom_DS))
    if ("Pred_SARIMA" %in% names(fp))
      fp$Pred_SARIMA <- as.numeric(unlist(fp$Pred_SARIMA))
    # Joindre seuils pour affichage mail
    fp <- dplyr::left_join(fp,
      seuils_df[, c("Nom_DS", "mois", "Seuil_Alerte", "Seuil_Epidemie")],
      by = c("Nom_DS", "mois"))
    fp
  }, error = function(e) { NULL })
}

# ─────────────────────────────────────────────────────────────────────────────
ui <- bslib::page_navbar(
  title = "Alerte Paludisme — Burkina Faso",
  theme = bslib::bs_theme(
    version = 5, bootswatch = "flatly",
    primary  = "#2C3E50", success = "#27AE60",
    warning  = "#F39C12", danger  = "#E74C3C"
  ),
  bg = "#2C3E50", inverse = TRUE,
  bslib::nav_panel(title = tagList(icon("gauge"),      " Tableau de bord"),   mod_dashboard_ui("dashboard")),
  bslib::nav_panel(title = tagList(icon("brain"),      " Modeles ML"),        mod_prediction_ui("prediction")),
  bslib::nav_panel(title = tagList(icon("map"),        " Carte des alertes"), mod_carte_ui("carte")),
  bslib::nav_panel(title = tagList(icon("chart-line"), " Predictions 6 mois"),mod_future_ui("future")),
  bslib::nav_panel(title = tagList(icon("envelope"),   " Alertes mail"),      mod_alertes_ui("alertes")),
  bslib::nav_panel(title = tagList(icon("file-word"),  " Rapport"),           mod_rapport_ui("rapport")),
  bslib::nav_spacer(),
  bslib::nav_item(
    actionButton("btn_train", "Lancer l analyse",
                 icon = icon("play"), class = "btn btn-success btn-sm")
  )
)

# ─────────────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  # ── Chargement Python ───────────────────────────────────────────────────────
  py <- tryCatch({
    reticulate::import_from_path("pipeline", path = PIPELINE_PATH)
  }, error = function(e) { NULL })

  # ── Valeurs réactives partagées ────────────────────────────────────────────
  rv <- reactiveValues(
    trained          = FALSE,
    loading          = FALSE,
    metrics          = NULL,
    best_name        = NULL,
    seuils           = NULL,
    test_alert       = NULL,
    future_pred      = NULL,
    feat_imp         = NULL,
    districts        = NULL,
    py               = py,           # ← py dans rv pour accès depuis les modules
    strategie_active = "YOUDEN_SAISONNIER"  # stratégie par défaut
  )

  # ── Chargement initial des données ─────────────────────────────────────────
  observe({
    req(rv$py)
    tryCatch({
      n <- rv$py$load_data(DATA_PATH)
      rv$districts <- rv$py$get_all_districts()
      showNotification(
        paste0("Donnees chargees : ", n, " observations"),
        type = "message", duration = 4)
    }, error = function(e) {
      showNotification(paste0("Erreur donnees : ", e$message),
                       type = "error", duration = 8)
    })
  })

  # ── Bouton Lancer l'analyse ────────────────────────────────────────────────
  observeEvent(input$btn_train, {
    req(rv$py)
    waiter::waiter_show(
      html  = tagList(waiter::spin_fading_circles(),
                      tags$h4("Entrainement en cours...",
                              style = "color:white; margin-top:20px;")),
      color = "rgba(44,62,80,0.90)"
    )

    tryCatch({
      withProgress(message = "Analyse en cours...", value = 0, {

        incProgress(0.1, detail = "Chargement des donnees")
        rv$py$load_data(DATA_PATH)

        incProgress(0.3, detail = "Entrainement KNN, RF, XGBoost, SARIMA")
        best <- rv$py$train_models()
        # train_models() calcule automatiquement les facteurs Youden
        # et les stocke dans _youden_factors côté Python

        incProgress(0.6, detail = "Calcul des metriques")
        rv$metrics   <- as.data.frame(do.call(rbind,
          lapply(rv$py$get_metrics(), as.data.frame)))
        rv$best_name <- best

        # Seuils
        s <- as.data.frame(do.call(rbind,
          lapply(rv$py$get_seuils(), as.data.frame)))
        s$mois             <- as.integer(unlist(s$mois))
        s$Baseline         <- as.numeric(unlist(s$Baseline))
        s$Seuil_Alerte     <- as.numeric(unlist(s$Seuil_Alerte))
        s$Seuil_Epidemie   <- as.numeric(unlist(s$Seuil_Epidemie))
        s$Nom_DS           <- as.character(unlist(s$Nom_DS))
        rv$seuils          <- s

        # test_alert — avec stratégie par défaut YOUDEN_SAISONNIER
        rv$py$apply_strategy(rv$strategie_active)
        ta <- as.data.frame(do.call(rbind,
          lapply(rv$py$get_test_alert(), as.data.frame)))
        ta$annee          <- as.integer(unlist(ta$annee))
        ta$mois           <- as.integer(unlist(ta$mois))
        ta$Pred           <- as.numeric(unlist(ta$Pred))
        ta$Cas_palu       <- as.numeric(unlist(ta$Cas_palu))
        ta$Baseline       <- as.numeric(unlist(ta$Baseline))
        ta$Seuil_Alerte   <- as.numeric(unlist(ta$Seuil_Alerte))
        ta$Seuil_Epidemie <- as.numeric(unlist(ta$Seuil_Epidemie))
        ta$Niveau_Alerte  <- as.character(unlist(ta$Niveau_Alerte))
        ta$Nom_DS         <- as.character(unlist(ta$Nom_DS))
        rv$test_alert     <- ta

        rv$feat_imp <- as.data.frame(do.call(rbind,
          lapply(rv$py$get_feature_importance(), as.data.frame)))

        # future_pred — avec stratégie par défaut
        incProgress(0.85, detail = "Predictions 6 mois")
        fp_raw <- rv$py$predict_future(6L)
        fp_updated <- rv$py$apply_strategy_future(fp_raw, rv$strategie_active)
        fp <- as.data.frame(do.call(rbind,
          lapply(fp_updated, function(x) as.data.frame(x, stringsAsFactors = FALSE))))
        fp$annee         <- as.integer(unlist(fp$annee))
        fp$mois          <- as.integer(unlist(fp$mois))
        fp$Pred_ML       <- as.numeric(unlist(fp$Pred_ML))
        fp$Niveau_Alerte <- as.character(unlist(fp$Niveau_Alerte))
        fp$Nom_DS        <- as.character(unlist(fp$Nom_DS))
        if ("Pred_SARIMA" %in% names(fp))
          fp$Pred_SARIMA <- as.numeric(unlist(fp$Pred_SARIMA))
        fp <- dplyr::left_join(fp,
          rv$seuils[, c("Nom_DS", "mois", "Seuil_Alerte", "Seuil_Epidemie")],
          by = c("Nom_DS", "mois"))
        rv$future_pred <- fp

        rv$trained <- TRUE
        rv$loading <- FALSE
        incProgress(1, detail = "Termine !")
      })

      waiter::waiter_hide()
      showNotification(
        paste0("✅ Analyse terminee ! Meilleur modele : ", best,
               " | Strategie : ", rv$strategie_active),
        type = "message", duration = 6)

    }, error = function(e) {
      waiter::waiter_hide()
      rv$loading <- FALSE
      message("=== TRACE COMPLETE ===")
      message(conditionMessage(e))
      showNotification(paste0("Erreur analyse : ", e$message),
                       type = "error", duration = 10)
    })
  })

  # ── Modules ────────────────────────────────────────────────────────────────
  mod_dashboard_server("dashboard",  rv)
  mod_prediction_server("prediction", rv)
  mod_carte_server("carte",          rv, SHAPEFILE_PATH)
  mod_future_server("future",        rv)
  mod_alertes_server("alertes",      rv)
  mod_rapport_server("rapport",      rv)
}

shinyApp(ui = ui, server = server)
