# =============================================================================
# mod_dashboard.R — Tableau de bord principal
# =============================================================================

# Facteur d'échelle pour l'affichage de l'incidence.
# À déplacer dans global.R pour être partagé entre modules.
# Actuel : 1000 (incidence pour 1000 habitants — standard SP-Palu / SIMR).
FACTEUR_ECHELLE <- 1000

mod_dashboard_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),
    # KPI Cards
    bslib::layout_columns(
      col_widths = c(3, 3, 3, 3),
      bslib::value_box(
        title    = "Districts surveillés",
        value    = textOutput(ns("n_districts")),
        showcase = icon("map-marker-alt"),
        theme    = "primary"
      ),
      bslib::value_box(
        title    = "Meilleur modèle",
        value    = textOutput(ns("best_model")),
        showcase = icon("trophy"),
        theme    = "success"
      ),
      bslib::value_box(
        title    = "Districts en alerte ROUGE",
        value    = textOutput(ns("n_rouge")),
        showcase = icon("exclamation-triangle"),
        theme    = "danger"
      ),
      bslib::value_box(
        title    = "Districts en vigilance",
        value    = textOutput(ns("n_vigilance")),
        showcase = icon("eye"),
        theme    = "warning"
      ),
      bslib::value_box(
        title    = "R² meilleur modèle",
        value    = textOutput(ns("r2_best")),
        showcase = icon("chart-bar"),
        theme    = "info"
      )
    ),
    br(),
    bslib::layout_columns(
      col_widths = c(8, 4),
      bslib::card(
        bslib::card_header(icon("chart-line"), " Évolution nationale de l'incidence du paludisme"),
        plotly::plotlyOutput(ns("plot_national"), height = "350px")
      ),
      bslib::card(
        bslib::card_header(icon("bell"), " Distribution des niveaux d'alerte"),
        plotly::plotlyOutput(ns("plot_alertes_pie"), height = "350px")
      )
    ),
    br(),
    bslib::card(
      bslib::card_header(icon("table"), " Seuils d'alerte par district"),
      DT::DTOutput(ns("tbl_seuils"))
    )
  )
}

mod_dashboard_server <- function(id, rv) {
  moduleServer(id, function(input, output, session) {
    
    output$n_districts <- renderText({
      if (!is.null(rv$districts)) length(rv$districts) else "70"
    })
    
    output$best_model <- renderText({
      if (!rv$trained) return("—")
      rv$best_name
    })
    
    output$n_rouge <- renderText({
      if (!rv$trained || is.null(rv$future_pred)) return("—")
      length(unique(rv$future_pred$Nom_DS[rv$future_pred$Niveau_Alerte == "EPIDEMIE"]))
    })

    output$n_vigilance <- renderText({
      if (!rv$trained || is.null(rv$future_pred)) return("—")
      length(unique(rv$future_pred$Nom_DS[rv$future_pred$Niveau_Alerte == "VIGILANCE"]))
    })
    
    output$r2_best <- renderText({
      if (!rv$trained || is.null(rv$metrics)) return("—")
      r2 <- rv$metrics$R2[rv$metrics$Modele == rv$best_name]
      if (length(r2)) paste0(round(r2, 3)) else "—"
    })
    
    output$plot_national <- plotly::renderPlotly({
      req(rv$trained, rv$test_alert)
      ta <- rv$test_alert
      nat <- ta |>
        dplyr::group_by(annee, mois) |>
        dplyr::summarise(
          Reel = mean(Cas_palu, na.rm = TRUE) * FACTEUR_ECHELLE,
          Pred = mean(Pred,     na.rm = TRUE) * FACTEUR_ECHELLE,
          .groups = "drop"
        ) |>
        dplyr::mutate(date = as.Date(paste(annee, mois, "01", sep = "-")))
      
      plotly::plot_ly(nat, x = ~date) |>
        plotly::add_lines(y = ~Reel, name = "Réel",
                          line = list(color = "#2980B9", width = 2)) |>
        plotly::add_lines(y = ~Pred, name = "Prédit",
                          line = list(color = "#E74C3C", width = 2, dash = "dash")) |>
        plotly::layout(
          xaxis  = list(title = ""),
          yaxis  = list(title = "Incidence nationale moyenne (/1000 hab.)"),
          legend = list(orientation = "h"),
          margin = list(t = 10)
        )
    })
    
    output$plot_alertes_pie <- plotly::renderPlotly({
      req(rv$trained, rv$future_pred)
      cnt    <- table(rv$future_pred$Niveau_Alerte)
      noms   <- names(cnt)
      colors <- c("EPIDEMIE" = "#E74C3C", "ALERTE" = "#F39C12", "VIGILANCE" = "#F7DC6F", "NORMAL" = "#27AE60")
      plotly::plot_ly(
        labels  = noms,
        values  = as.numeric(cnt),
        type    = "pie",
        marker  = list(colors = unname(colors[noms]))
      ) |>
        plotly::layout(showlegend = TRUE, margin = list(t = 10))
    })
    
    output$tbl_seuils <- DT::renderDT({
      req(rv$trained, rv$seuils)
      s <- rv$seuils
      # Copie d'affichage : mise à l'échelle des colonnes d'incidence.
      cols_num <- c("Baseline", "Seuil_Vigilance", "Seuil_Alerte", "Seuil_Epidemie")
      s_display <- s
      s_display[cols_num] <- lapply(
        s_display[cols_num],
        function(x) round(x * FACTEUR_ECHELLE, 2)
      )
      DT::datatable(
        s_display[, c("Nom_DS", "mois", cols_num)],
        colnames  = c("District", "Mois",
                      "Baseline (/1000 hab.)",
                      "Seuil Vigilance Q1 (/1000 hab.)",
                      "Seuil Alerte Q2 (/1000 hab.)",
                      "Seuil Épidémie Q3 (/1000 hab.)"),
        options   = list(pageLength = 12, scrollX = TRUE),
        rownames  = FALSE
      )
    })
  })
}