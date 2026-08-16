# =============================================================================
# mod_future.R — Prédictions 6 mois (avec propagation stratégie de seuil)
# =============================================================================

# Facteur d'échelle pour l'affichage de l'incidence.
# À déplacer dans global.R pour être partagé entre modules.
# Actuel : 1000 (incidence pour 1000 habitants — standard SP-Palu / SIMR).
FACTEUR_ECHELLE <- 1000

mod_future_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),
    bslib::layout_columns(
      col_widths = c(4, 8),
      bslib::card(
        bslib::card_header(icon("filter"), " Filtres"),
        shinyWidgets::pickerInput(
          ns("district_sel"), "District :",
          choices = character(0), multiple = FALSE,
          options = shinyWidgets::pickerOptions(liveSearch = TRUE)
        ),

        # Indicateur stratégie active
        uiOutput(ns("badge_strategie")),

        br(),
        bslib::card(
          bslib::card_header("Résumé alertes — 6 mois"),
          tableOutput(ns("tbl_summary_future"))
        )
      ),
      bslib::card(
        bslib::card_header(icon("chart-area"), " Prédictions par district — 6 mois"),
        plotly::plotlyOutput(ns("plot_future"), height = "380px")
      )
    ),
    br(),
    bslib::card(
      bslib::card_header(icon("table"), " Tableau complet des prédictions futures"),
      DT::DTOutput(ns("tbl_future"))
    )
  )
}

mod_future_server <- function(id, rv) {
  moduleServer(id, function(input, output, session) {

    # Mettre à jour la liste des districts
    observe({
      req(rv$districts)
      shinyWidgets::updatePickerInput(
        session, "district_sel",
        choices  = rv$districts,
        selected = rv$districts[1]
      )
    })

    # Badge stratégie active
    output$badge_strategie <- renderUI({
      strat <- if (!is.null(rv$strategie_active)) rv$strategie_active else "YOUDEN_SAISONNIER"
      label <- switch(strat,
        "Q3"                = "① Seuil Q3",
        "YOUDEN_GLOBAL"     = "② Youden global",
        "YOUDEN_SAISONNIER" = "③ Youden saisonnier"
      )
      couleur <- switch(strat,
        "Q3"                = "#F39C12",
        "YOUDEN_GLOBAL"     = "#2980B9",
        "YOUDEN_SAISONNIER" = "#27AE60"
      )
      tags$div(
        style = paste0(
          "background:", couleur, "20; border-left:3px solid ", couleur, ";",
          "padding:6px 10px; border-radius:4px; margin-top:8px; font-size:12px;"
        ),
        icon("check-circle", style = paste0("color:", couleur, ";")),
        tags$strong(paste0(" Stratégie active : ", label))
      )
    })

    # Graphique futur (réactif sur future_pred)
    output$plot_future <- plotly::renderPlotly({
      req(rv$trained, rv$future_pred, input$district_sel)
      fp <- rv$future_pred
      ds <- fp[fp$Nom_DS == input$district_sel, ]
      ds$label_mois <- paste0(ds$annee, "/M", sprintf("%02d", ds$mois))

      cols     <- c("NORMAL"="#27AE60","VIGILANCE"="#F7DC6F","ALERTE"="#F39C12","EPIDEMIE"="#E74C3C")
      bar_cols <- unname(cols[ds$Niveau_Alerte])

      plotly::plot_ly(
        ds, x = ~label_mois, y = ~Pred_ML * FACTEUR_ECHELLE,
        type      = "bar",
        marker    = list(color = bar_cols),
        text      = ~paste0("Alerte : ", Niveau_Alerte,
                             "<br>Incidence : ",
                             format(round(Pred_ML * FACTEUR_ECHELLE, 2), nsmall = 2),
                             " /1000 hab."),
        hoverinfo = "text"
      ) |>
        plotly::layout(
          xaxis  = list(title = ""),
          yaxis  = list(title = "Incidence prédite (/1000 hab.)"),
          margin = list(t = 20)
        )
    })

    output$tbl_summary_future <- renderTable({
      req(rv$trained, rv$future_pred)
      fp  <- rv$future_pred
      cnt <- table(fp$Niveau_Alerte)
      data.frame(Niveau = names(cnt), `Districts-mois` = as.integer(cnt), check.names = FALSE)
    }, striped = TRUE, hover = TRUE)

    output$tbl_future <- DT::renderDT({
      req(rv$trained, rv$future_pred)
      fp <- rv$future_pred
      # Copie d'affichage : on met à l'échelle les colonnes numériques d'incidence
      # (les dataframes originaux et la logique d'alerte ne sont pas touchés).
      cols_incidence <- intersect(
        c("Pred_ML", "Pred_ML2", "Pred_SARIMA",
          "Seuil_Vigilance", "Seuil_Alerte", "Seuil_Epidemie"),
        names(fp)
      )
      fp_display <- fp
      fp_display[cols_incidence] <- lapply(
        fp_display[cols_incidence],
        function(x) round(x * FACTEUR_ECHELLE, 2)
      )
      DT::datatable(
        fp_display[order(fp_display$Niveau_Alerte, fp_display$annee, fp_display$mois), ],
        rownames = FALSE,
        options  = list(pageLength = 15, scrollX = TRUE),
        filter   = "top",
        caption  = htmltools::tags$caption(
          style = "caption-side: top; text-align: left; font-style: italic;",
          "Toutes les incidences et seuils sont exprimés pour 1000 habitants."
        )
      ) |>
        DT::formatStyle(
          "Niveau_Alerte",
          backgroundColor = DT::styleEqual(
            c("NORMAL","VIGILANCE","ALERTE","EPIDEMIE"),
            c("#D5F5E3","#FFFDE7","#FAD7A0","#FADBD8")
          ),
          fontWeight = "bold"
        )
    })
  })
}
