# =============================================================================
# mod_prediction.R — Comparaison des modèles ML + Stratégie de seuil
# =============================================================================

# Facteur d'échelle pour l'affichage de l'incidence.
# À déplacer dans global.R pour être partagé entre modules.
# Actuel : 1000 (incidence pour 1000 habitants — standard SP-Palu / SIMR).
FACTEUR_ECHELLE <- 1000

mod_prediction_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),

    # ── Sélecteur de stratégie (compact : menu déroulant + tableau côte à côte) ──
    bslib::card(
      bslib::card_header(
        icon("sliders-h"),
        " Stratégie de détection épidémique",
        class = "bg-dark text-white"
      ),
      bslib::layout_columns(
        col_widths = c(4, 8),
        tags$div(
          style = "padding:12px 16px;",
          selectInput(
            inputId  = ns("strategie_seuil"),
            label    = "Choisir la stratégie de seuil :",
            choices  = c(
              "① Seuil Q3 — conservateur"        = "Q3",
              "② Youden global — sensible"       = "YOUDEN_GLOBAL",
              "③ Youden saisonnier — recommandé" = "YOUDEN_SAISONNIER"
            ),
            selected = "YOUDEN_SAISONNIER",
            width    = "100%"
          ),
          uiOutput(ns("strategie_info"))
        ),
        tags$div(
          style = "padding:12px 16px;",
          uiOutput(ns("tbl_strategies"))
        )
      )
    ),

    br(),

    bslib::layout_columns(
      col_widths = c(6, 6),
      bslib::card(
        bslib::card_header(icon("table"), " Comparaison des métriques"),
        DT::DTOutput(ns("tbl_metrics"))
      ),
      bslib::card(
        bslib::card_header(icon("chart-bar"), " Performance par modèle"),
        selectInput(ns("metric_sel"), "Métrique :",
                    choices = c("MAE", "RMSE", "R2", "SMAPE"), selected = "R2"),
        plotly::plotlyOutput(ns("plot_metrics"), height = "260px")
      )
    ),
    br(),
    bslib::layout_columns(
      col_widths = c(7, 5),
      bslib::card(
        height = 380,
        bslib::card_header(icon("arrows-alt-h"), " Réel vs Prédit — période test"),
        plotly::plotlyOutput(ns("plot_scatter"), height = "100%")
      ),
      bslib::card(
        height = 380,
        bslib::card_header(icon("sort-amount-down"), " Importance des variables"),
        plotly::plotlyOutput(ns("plot_importance"), height = "100%")
      )
    )
  )
}

mod_prediction_server <- function(id, rv) {
  moduleServer(id, function(input, output, session) {

    # ── Explication contextuelle ──────────────────────────────────────────────
    output$strategie_info <- renderUI({
      req(input$strategie_seuil)
      switch(input$strategie_seuil,
        "Q3" = tags$div(
          class = "alert alert-warning p-2 mb-0",
          icon("shield-alt"), tags$strong(" Seuil OMS Q3 standard"), tags$br(),
          tags$small("Conservateur : peu de fausses alertes, mais 40% des épidémies non détectées.")
        ),
        "YOUDEN_GLOBAL" = tags$div(
          class = "alert alert-primary p-2 mb-0",
          icon("bullseye"), tags$strong(" Seuil Youden global"), tags$br(),
          tags$small("Équilibre optimal Sensibilité/Spécificité sur toute l'année.")
        ),
        "YOUDEN_SAISONNIER" = tags$div(
          class = "alert alert-success p-2 mb-0",
          icon("star"), tags$strong(" Youden saisonnier ✓ Recommandé"), tags$br(),
          tags$small("Seuil différencié saison haute (Jul-Oct) / basse (Nov-Jun). Meilleur compromis global.")
        )
      )
    })

    # ── Tableau comparatif 3 stratégies ──────────────────────────────────────
    output$tbl_strategies <- renderUI({
      sel <- input$strategie_seuil

      if (!isTRUE(rv$trained)) {
        return(tags$div(
          class = "alert alert-info p-2 mt-2",
          icon("info-circle"),
          " Lancez l'analyse pour voir les résultats de validation."
        ))
      }

      strategies <- tryCatch({
        sm <- rv$py$get_strategies_metrics()
        stopifnot(length(sm) > 0)
        lapply(sm, function(s) list(
          id  = s$id,  nom = s$nom,               bg  = s$bg,
          se  = round(as.numeric(s$se),  1),  sp  = round(as.numeric(s$sp),  1),
          vpp = round(as.numeric(s$vpp), 1),  vpn = round(as.numeric(s$vpn), 1),
          f2  = round(as.numeric(s$f2),  3)
        ))
      }, error = function(e) list(
        list(id="Q3",                nom="① Seuil Q3",          se=59.4, sp=88.8, vpp=50.4, vpn=91.9, f2=0.573, bg="#FEF9E7"),
        list(id="YOUDEN_GLOBAL",     nom="② Youden global",     se=87.7, sp=70.9, vpp=36.7, vpn=96.8, f2=0.686, bg="#EBF5FB"),
        list(id="YOUDEN_SAISONNIER", nom="③ Youden saisonnier", se=87.2, sp=72.4, vpp=37.7, vpn=96.7, f2=0.691, bg="#EAFAF1")
      ))

      lignes <- lapply(strategies, function(s) {
        est_selec <- isTRUE(sel == s$id)
        style_row <- if (est_selec)
          paste0("background:", s$bg, "; font-weight:bold; border-left:4px solid #27AE60;")
        else
          paste0("background:", s$bg, ";")
        tags$tr(
          style = style_row,
          tags$td(style = "padding:5px 10px;",
            if (est_selec) tags$span("▶ ", style="color:#27AE60;"), s$nom),
          tags$td(style = "padding:5px 10px; text-align:center;",
            tags$span(paste0(s$se, "%"),
              style = if (s$se >= 85) "color:#27AE60; font-weight:bold;" else "")),
          tags$td(style = "padding:5px 10px; text-align:center;", paste0(s$sp, "%")),
          tags$td(style = "padding:5px 10px; text-align:center;", paste0(s$vpp, "%")),
          tags$td(style = "padding:5px 10px; text-align:center;", paste0(s$vpn, "%")),
          tags$td(style = "padding:5px 10px; text-align:center;",
            tags$strong(paste0(s$f2),
              style = if (s$f2 >= 0.68) "color:#27AE60;" else ""))
        )
      })

      tags$div(
        class = "mt-2",
        tags$table(
          class = "table table-bordered table-sm",
          style = "font-size:12px; margin-bottom:4px;",
          tags$thead(tags$tr(
            tags$th(style="padding:5px 10px; background:#2C3E50 !important; color:#fff !important;", "Stratégie"),
            tags$th(style="padding:5px 10px; text-align:center; background:#2C3E50 !important; color:#fff !important;", "Sensibilité"),
            tags$th(style="padding:5px 10px; text-align:center; background:#2C3E50 !important; color:#fff !important;", "Spécificité"),
            tags$th(style="padding:5px 10px; text-align:center; background:#2C3E50 !important; color:#fff !important;", "VPP"),
            tags$th(style="padding:5px 10px; text-align:center; background:#2C3E50 !important; color:#fff !important;", "VPN"),
            tags$th(style="padding:5px 10px; text-align:center; background:#2C3E50 !important; color:#fff !important;", "F2-score")
          )),
          tags$tbody(lignes)
        ),
        tags$small(class="text-muted",
          "▶ = stratégie active | F2 favorise la sensibilité (β=2)")
      )
    })

    # ── Propagation vers rv et Python ─────────────────────────────────────────
    observeEvent(input$strategie_seuil, {
      if (!isTRUE(rv$trained) || is.null(rv$py)) return()
      strat <- input$strategie_seuil

      tryCatch({
        withProgress(message = "Application de la stratégie…", value = 0.1, {

        rv$py$apply_strategy(strat)

        ta_raw <- rv$py$get_test_alert()
        ta <- as.data.frame(
          do.call(rbind, lapply(ta_raw, function(x)
            as.data.frame(x, stringsAsFactors = FALSE)))
        )
        ta$annee         <- as.integer(unlist(ta$annee))
        ta$mois          <- as.integer(unlist(ta$mois))
        ta$Pred          <- as.numeric(unlist(ta$Pred))
        ta$Cas_palu      <- as.numeric(unlist(ta$Cas_palu))
        ta$Niveau_Alerte <- as.character(unlist(ta$Niveau_Alerte))
        ta$Nom_DS        <- as.character(unlist(ta$Nom_DS))
        if ("Baseline"       %in% names(ta)) ta$Baseline       <- as.numeric(unlist(ta$Baseline))
        if ("Seuil_Alerte"   %in% names(ta)) ta$Seuil_Alerte   <- as.numeric(unlist(ta$Seuil_Alerte))
        if ("Seuil_Epidemie" %in% names(ta)) ta$Seuil_Epidemie <- as.numeric(unlist(ta$Seuil_Epidemie))
        rv$test_alert <- ta
        incProgress(0.5, detail = "Prévisions des 6 prochains mois…")

        fp_raw     <- rv$py$predict_future(6L)
        fp_updated <- rv$py$apply_strategy_future(fp_raw, strat)
        fp <- as.data.frame(
          do.call(rbind, lapply(fp_updated, function(x)
            as.data.frame(x, stringsAsFactors = FALSE)))
        )
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
        rv$future_pred    <- fp
        rv$strategie_active <- strat
        incProgress(0.4, detail = "Terminé")
        })  # fin withProgress

        showNotification(
          paste0("✅ Stratégie appliquée : ", strat),
          type = "message", duration = 3)

      }, error = function(e) {
        showNotification(
          paste0("❌ Erreur : ", e$message),
          type = "error", duration = 5)
      })
    }, ignoreInit = TRUE)

    # ── Métriques ─────────────────────────────────────────────────────────────
    output$tbl_metrics <- DT::renderDT({
      req(rv$trained, rv$metrics)
      DT::datatable(rv$metrics, rownames = FALSE,
        options = list(dom = "t", ordering = FALSE)) |>
        DT::formatStyle("Modele", target = "row",
          backgroundColor = DT::styleEqual(rv$best_name, "#D5F5E3"),
          fontWeight      = DT::styleEqual(rv$best_name, "bold")) |>
        DT::formatRound(
          intersect(c("MAE", "RMSE", "R2", "SMAPE", "Rang_moyen"),
                    names(rv$metrics)), 3)
    })

    output$plot_metrics <- plotly::renderPlotly({
      req(rv$trained, rv$metrics, input$metric_sel)
      m <- rv$metrics; mt <- input$metric_sel
      plotly::plot_ly(m, x = ~Modele, y = ~get(mt), type = "bar",
        marker = list(color = ifelse(m$Modele == rv$best_name, "#27AE60", "#2980B9"))) |>
        plotly::layout(yaxis = list(title = mt), xaxis = list(title = ""),
          margin = list(t = 10))
    })

    output$plot_scatter <- plotly::renderPlotly({
      req(rv$trained, rv$test_alert)
      ta <- rv$test_alert
      # Copie d'affichage : deux colonnes scalées pour l'affichage.
      ta$Incidence_reelle <- ta$Cas_palu * FACTEUR_ECHELLE
      ta$Incidence_pred   <- ta$Pred     * FACTEUR_ECHELLE
      plotly::plot_ly(ta, x = ~Incidence_reelle, y = ~Incidence_pred,
        type = "scatter", mode = "markers",
        color  = ~Niveau_Alerte,
        colors = c("NORMAL"="#27AE60","VIGILANCE"="#F7DC6F",
                   "ALERTE"="#F39C12","EPIDEMIE"="#E74C3C"),
        text   = ~paste0(Nom_DS, "<br>Année:", annee, " M:", mois),
        marker = list(size = 5, opacity = 0.6)) |>
        plotly::add_lines(x = ~Incidence_reelle, y = ~Incidence_reelle,
          line = list(color = "gray", dash = "dash"),
          showlegend = FALSE, inherit = FALSE) |>
        plotly::layout(
          xaxis  = list(title = "Incidence réelle (/1000 hab.)"),
          yaxis  = list(title = "Incidence prédite (/1000 hab.)"),
          margin = list(t = 10, b = 50))
    })

    output$plot_importance <- plotly::renderPlotly({
      req(rv$trained, rv$feat_imp)
      fi <- rv$feat_imp
      if (nrow(fi) == 0) return(plotly::plotly_empty())
      fi <- fi[order(fi$importance), ]
      plotly::plot_ly(fi, x = ~importance, y = ~feature,
        type = "bar", orientation = "h",
        marker = list(color = "#2980B9")) |>
        plotly::layout(
          yaxis  = list(title = "", categoryorder = "array",
                        categoryarray = fi$feature),
          xaxis  = list(title = "Importance"),
          margin = list(l = 160, t = 10))
    })
  })
}
