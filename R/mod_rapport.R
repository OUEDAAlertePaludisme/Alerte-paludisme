# =============================================================================
# mod_rapport.R — Génération rapport .docx avec Anthropic Claude API
# Corrections : timeout API, bug seuils_tbl, progress robuste
# =============================================================================

mod_rapport_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),
    bslib::layout_columns(
      col_widths = c(4, 8),

      bslib::card(
        bslib::card_header(icon("cog"), " Configuration"),
        textInput(ns("api_key"), "Clé API Anthropic :",
                  placeholder = "sk-ant-..."),
        tags$small(class = "text-muted",
                   "Disponible sur console.anthropic.com"),
        tags$a(href = "https://console.anthropic.com/settings/keys",
               target = "_blank",
               icon("external-link-alt"), " Obtenir une clé API"),
        hr(),
        textInput(ns("auteur"), "Auteur :", value = "OUEDA"),
        textInput(ns("institution"), "Institution :",
                  value = "Ministère de la Santé — Burkina Faso"),
        hr(),
        checkboxGroupInput(
          ns("sections"), "Sections à inclure :",
          choices  = c("Résumé exécutif (IA)"    = "resume",
                       "Comparaison des modèles"  = "modeles",
                       "Prédictions 6 mois"       = "predictions",
                       "Alertes par district"     = "alertes",
                       "Recommandations (IA)"     = "recommandations"),
          selected = c("resume", "modeles", "predictions",
                       "alertes", "recommandations")
        ),
        hr(),
        actionButton(ns("btn_generate"), "Générer le rapport",
                     icon  = icon("file-word"),
                     class = "btn-primary w-100"),
        br(), br(),
        uiOutput(ns("download_btn"))
      ),

      bslib::card(
        bslib::card_header(icon("eye"), " Aperçu du rapport"),
        uiOutput(ns("rapport_preview"))
      )
    )
  )
}

mod_rapport_server <- function(id, rv) {
  moduleServer(id, function(input, output, session) {

    rapport_path <- reactiveVal(NULL)

    # ── Nettoyage Markdown ────────────────────────────────────────────────────
    clean_markdown <- function(text) {
      text <- gsub("^#+\\s*", "", text, perl = TRUE)
      text <- gsub("\\*\\*([^*]+)\\*\\*", "\\1", text)
      text <- gsub("\\*([^*]+)\\*", "\\1", text)
      text <- gsub("^[-*]\\s+", "", text, perl = TRUE)
      text <- gsub("^---+$", "", text, perl = TRUE)
      text <- gsub("\n{3,}", "\n\n", text)
      text <- trimws(text)
      text
    }

    # ── Appel Anthropic Claude API (avec timeout) ─────────────────────────────
    claude_interpret <- function(prompt, api_key) {
      tryCatch({
        res <- httr::POST(
          url  = "https://api.anthropic.com/v1/messages",
          httr::add_headers(
            `x-api-key`         = api_key,
            `anthropic-version` = "2023-06-01",
            `content-type`      = "application/json"
          ),
          httr::timeout(30),   # ← évite le blocage infini
          body = jsonlite::toJSON(list(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 1000L,
            messages   = list(
              list(role = "user", content = prompt)
            )
          ), auto_unbox = TRUE)
        )

        if (httr::status_code(res) != 200) {
          err <- httr::content(res, as = "parsed")
          return(paste0("Erreur API Claude : ", err$error$message))
        }

        content <- httr::content(res, as = "parsed")
        clean_markdown(content$content[[1]]$text)

      }, error = function(e) {
        paste0("Erreur connexion Claude : ", e$message)
      })
    }

    # ── Génération du rapport ─────────────────────────────────────────────────
    observeEvent(input$btn_generate, {
      req(rv$trained, rv$metrics, rv$best_name, rv$future_pred, rv$test_alert)

      withProgress(message = "Génération du rapport...", value = 0, {

        metrics <- rv$metrics
        best    <- rv$best_name
        fp      <- rv$future_pred
        use_ai  <- nzchar(trimws(input$api_key))

        texte_resume  <- ""
        texte_modeles <- ""
        texte_reco    <- ""

        if (use_ai && "resume" %in% input$sections) {
          incProgress(0.1, detail = "Rédaction résumé exécutif (IA)...")
          n_rouge  <- length(unique(fp$Nom_DS[fp$Niveau_Alerte == "EPIDEMIE"]))
          n_orange <- length(unique(fp$Nom_DS[fp$Niveau_Alerte == "ALERTE"]))
          texte_resume <- claude_interpret(paste0(
            "Tu es un épidémiologiste expert en paludisme au Burkina Faso. ",
            "Rédige un résumé exécutif professionnel en 5 phrases maximum. ",
            "IMPORTANT : réponds uniquement en texte brut, sans aucun titre, ",
            "sans Markdown, sans # ni ** ni tirets. Juste des phrases. ",
            "Le meilleur modèle est ", best,
            " avec R²=", round(metrics$R2[metrics$Modele == best], 4),
            ", MAE=", round(metrics$MAE[metrics$Modele == best], 2), ". ",
            "Sur les 6 prochains mois, ", n_rouge,
            " districts sont en alerte ROUGE, ", n_orange, " en ORANGE et ",
            length(unique(fp$Nom_DS[fp$Niveau_Alerte == "VIGILANCE"])), " en VIGILANCE. ",
            "Rédige en français, ton professionnel."
          ), input$api_key)
        }

        if (use_ai && "modeles" %in% input$sections) {
          incProgress(0.2, detail = "Interprétation des modèles (IA)...")
          texte_modeles <- claude_interpret(paste0(
            "Interprète ces résultats de modèles ML pour la prédiction du paludisme. ",
            "IMPORTANT : réponds en texte brut uniquement, sans Markdown, ",
            "sans # ni ** ni tirets ni listes. Juste 3 phrases continues. ",
            "SARIMA: MAE=", metrics$MAE[metrics$Modele == "SARIMA"],
            " R2=", metrics$R2[metrics$Modele == "SARIMA"], ". ",
            "KNN: MAE=", metrics$MAE[metrics$Modele == "KNN"],
            " R2=", metrics$R2[metrics$Modele == "KNN"], ". ",
            "RandomForest: MAE=", metrics$MAE[metrics$Modele == "RandomForest"],
            " R2=", metrics$R2[metrics$Modele == "RandomForest"], ". ",
            "XGBoost: MAE=", metrics$MAE[metrics$Modele == "XGBoost"],
            " R2=", metrics$R2[metrics$Modele == "XGBoost"], ". ",
            "Le meilleur est ", best, ". En français."
          ), input$api_key)
        }

        if (use_ai && "recommandations" %in% input$sections) {
          incProgress(0.3, detail = "Rédaction recommandations (IA)...")
          ds_rouge <- unique(fp$Nom_DS[fp$Niveau_Alerte == "EPIDEMIE"])
          texte_reco <- claude_interpret(paste0(
            "Tu es épidémiologiste spécialiste du paludisme au Burkina Faso. ",
            "IMPORTANT : réponds en texte brut uniquement, sans Markdown, ",
            "sans # ni ** ni tirets. Numérote simplement 1. 2. 3. 4. ",
            "Districts en alerte ROUGE : ",
            paste(head(ds_rouge, 10), collapse = ", "), ". ",
            "Rédige 4 recommandations opérationnelles concrètes. ",
            "En français, ton directif et concis."
          ), input$api_key)
        }

        incProgress(0.5, detail = "Mise en page du document Word...")

        doc <- officer::read_docx()
        num <- 0

        # Page de garde
        doc <- officer::body_add_par(doc,
          "SYSTÈME D'ALERTE PRÉCOCE DU PALUDISME", style = "heading 1")
        doc <- officer::body_add_par(doc,
          "Burkina Faso — 70 Districts Sanitaires", style = "heading 2")
        doc <- officer::body_add_par(doc,
          paste0("Auteur : ", input$auteur,
                 "  |  Institution : ", input$institution,
                 "  |  Date : ", format(Sys.Date(), "%d/%m/%Y")),
          style = "Normal")
        doc <- officer::body_add_par(doc, "", style = "Normal")
        doc <- officer::body_add_par(doc, "", style = "Normal")

        # Résumé exécutif
        if ("resume" %in% input$sections) {
          num <- num + 1
          doc <- officer::body_add_par(doc,
            paste0(num, ". Résumé exécutif"), style = "heading 2")
          doc <- officer::body_add_par(doc,
            if (nzchar(texte_resume)) texte_resume
            else paste0(
              "Ce rapport présente les résultats du système de prédiction ",
              "du paludisme sur 70 districts sanitaires du Burkina Faso. ",
              "Le meilleur modèle sélectionné est ", best,
              " (R²=", round(metrics$R2[metrics$Modele == best], 4), ")."
            ),
            style = "Normal")
        }

        # Comparaison modèles
        if ("modeles" %in% input$sections) {
          num <- num + 1
          doc <- officer::body_add_par(doc,
            paste0(num, ". Comparaison des modèles"), style = "heading 2")
          if (nzchar(texte_modeles))
            doc <- officer::body_add_par(doc, texte_modeles, style = "Normal")
          ft <- flextable::flextable(metrics) |>
            flextable::set_header_labels(
              Modele = "Modèle", R2 = "R²",
              SMAPE = "SMAPE (%)", Rang_moyen = "Rang moyen") |>
            flextable::bg(
              i  = which(metrics$Modele == best),
              bg = "#D5F5E3", part = "body"
            ) |>
            flextable::bold(i = which(metrics$Modele == best)) |>
            flextable::theme_booktabs() |>
            flextable::autofit()
          doc <- flextable::body_add_flextable(doc, ft)
        }

        # Prédictions
        if ("predictions" %in% input$sections) {
          num <- num + 1
          doc <- officer::body_add_par(doc,
            paste0(num, ". Prédictions 6 mois"), style = "heading 2")
          rouge_ds      <- unique(fp$Nom_DS[fp$Niveau_Alerte == "EPIDEMIE"])
          orange_ds     <- unique(fp$Nom_DS[fp$Niveau_Alerte == "ALERTE"])
          vigilance_ds  <- unique(fp$Nom_DS[fp$Niveau_Alerte == "VIGILANCE"])
          doc <- officer::body_add_par(doc,
            paste0("Districts en alerte ROUGE : ", length(rouge_ds),
                   " | Districts en alerte ORANGE : ", length(orange_ds),
                   " | Districts en VIGILANCE : ", length(vigilance_ds)),
            style = "Normal")
          if (length(rouge_ds) > 0)
            doc <- officer::body_add_par(doc,
              paste0("Districts ROUGE : ", paste(rouge_ds, collapse = ", ")),
              style = "Normal")
          if (length(vigilance_ds) > 0)
            doc <- officer::body_add_par(doc,
              paste0("Districts VIGILANCE : ", paste(vigilance_ds, collapse = ", ")),
              style = "Normal")
        }

        # Tableau des alertes par district — CORRIGÉ
        if ("alertes" %in% input$sections) {
          num <- num + 1
          doc <- officer::body_add_par(doc,
            paste0(num, ". Tableau des alertes par district"),
            style = "heading 2")
          doc <- officer::body_add_par(doc,
            paste0("Le tableau ci-dessous présente les seuils d'alerte ",
                   "et le niveau de référence (baseline) pour les 70 districts ",
                   "sanitaires du Burkina Faso."),
            style = "Normal")

          # CORRECTION : utiliser rv$test_alert qui contient Niveau_Alerte
          # et sélectionner uniquement les colonnes disponibles
          ta <- rv$test_alert
          cols_dispo <- intersect(
            c("Nom_DS", "Baseline", "Seuil_Vigilance", "Seuil_Alerte", "Seuil_Epidemie", "Niveau_Alerte"),
            names(ta)
          )
          seuils_tbl <- ta[, cols_dispo]

          # Résumer par district (dernier mois disponible)
          seuils_tbl <- seuils_tbl[!duplicated(seuils_tbl$Nom_DS, fromLast = TRUE), ]
          names(seuils_tbl) <- c("District", "Baseline",
                                  "Seuil Vigilance", "Seuil Alerte", "Seuil Épidémie", "Niveau")[
                                    seq_along(cols_dispo)]

          idx_rouge      <- which(seuils_tbl$Niveau == "EPIDEMIE")
          idx_orange     <- which(seuils_tbl$Niveau == "ALERTE")
          idx_vigilance  <- which(seuils_tbl$Niveau == "VIGILANCE")
          idx_vert       <- which(seuils_tbl$Niveau == "NORMAL")

          ft_seuils <- flextable::flextable(seuils_tbl) |>
            flextable::theme_booktabs() |>
            flextable::autofit() |>
            flextable::bold(j = "Niveau") |>
            flextable::color(i = idx_rouge,     j = "Niveau", color = "#E74C3C") |>
            flextable::color(i = idx_orange,    j = "Niveau", color = "#F39C12") |>
            flextable::color(i = idx_vigilance, j = "Niveau", color = "#D4AC0D") |>
            flextable::color(i = idx_vert,      j = "Niveau", color = "#27AE60") |>
            flextable::bg(i = idx_rouge,     bg = "#FADBD8", part = "body") |>
            flextable::bg(i = idx_orange,    bg = "#FAD7A0", part = "body") |>
            flextable::bg(i = idx_vigilance, bg = "#FFFDE7", part = "body") |>
            flextable::bg(i = idx_vert,      bg = "#D5F5E3", part = "body")

          doc <- flextable::body_add_flextable(doc, ft_seuils)
        }

        # Recommandations
        if ("recommandations" %in% input$sections) {
          num <- num + 1
          doc <- officer::body_add_par(doc,
            paste0(num, ". Recommandations"), style = "heading 2")
          doc <- officer::body_add_par(doc,
            if (nzchar(texte_reco)) texte_reco
            else paste0(
              "1. Renforcer la surveillance épidémiologique dans les districts en alerte ROUGE.\n",
              "2. Vérifier et reconstituer les stocks d'ACT et de TDR dans les zones à risque.\n",
              "3. Mobiliser les équipes de riposte rapide avant le pic de transmission.\n",
              "4. Intensifier la distribution de moustiquaires imprégnées (MILDA).\n",
              "5. Assurer un suivi hebdomadaire des cas pendant toute la saison de transmission."
            ),
            style = "Normal")
        }

        incProgress(0.9, detail = "Enregistrement du fichier...")
        path <- tempfile(fileext = ".docx")
        print(doc, target = path)
        rapport_path(path)
        incProgress(1, detail = "Terminé !")

      }) # ← ferme withProgress

      showNotification("✅ Rapport généré avec succès !", type = "message", duration = 5)

    }) # ← ferme observeEvent

    output$download_btn <- renderUI({
      req(rapport_path())
      downloadButton(session$ns("download"),
                     "⬇️ Télécharger le rapport (.docx)",
                     class = "btn-success w-100")
    })

    output$download <- downloadHandler(
      filename = function() paste0("rapport_paludisme_", Sys.Date(), ".docx"),
      content  = function(file) file.copy(rapport_path(), file)
    )

    output$rapport_preview <- renderUI({
      if (!rv$trained) {
        return(tags$div(
          class = "text-center text-muted p-4",
          icon("info-circle", style = "font-size:2em;"), br(), br(),
          "Lancez d'abord l'analyse avec le bouton ",
          tags$strong("\"Lancer l'analyse\""), " en haut à droite."
        ))
      }
      tagList(
        tags$h5(icon("check-circle"), " L'analyse est prête"),
        tags$table(
          class = "table table-sm table-bordered",
          tags$tr(tags$td(tags$strong("Meilleur modèle")),
                  tags$td(rv$best_name)),
          tags$tr(tags$td(tags$strong("R²")),
                  tags$td(round(rv$metrics$R2[rv$metrics$Modele == rv$best_name], 4))),
          tags$tr(tags$td(tags$strong("MAE")),
                  tags$td(round(rv$metrics$MAE[rv$metrics$Modele == rv$best_name], 2))),
          tags$tr(
            tags$td(tags$strong("Districts ROUGE (6 mois)")),
            tags$td(length(unique(
              rv$future_pred$Nom_DS[rv$future_pred$Niveau_Alerte == "EPIDEMIE"])))
          ),
          tags$tr(
            tags$td(tags$strong("Districts ORANGE (6 mois)")),
            tags$td(length(unique(
              rv$future_pred$Nom_DS[rv$future_pred$Niveau_Alerte == "ALERTE"])))
          ),
          tags$tr(
            tags$td(tags$strong("Districts JAUNE (6 mois)")),
            tags$td(length(unique(
              rv$future_pred$Nom_DS[rv$future_pred$Niveau_Alerte == "VIGILANCE"])))
          )
        ),
        hr(),
        tags$h6("Sections sélectionnées :"),
        tags$ul(
          tags$li("Page de garde (auteur, institution, date)"),
          tags$li("Résumé exécutif"),
          tags$li("Tableau comparatif des modèles ML"),
          tags$li("Prédictions 6 mois — districts en alerte"),
          tags$li("Tableau des alertes par district"),
          tags$li("Recommandations opérationnelles")
        ),
        hr(),
        tags$small(class = "text-muted",
          icon("info-circle"), " IA propulsée par ",
          tags$strong("Anthropic Claude Haiku"),
          " — console.anthropic.com"
        )
      )
    }) # ← ferme output$rapport_preview

  }) # ← ferme moduleServer
}    # ← ferme mod_rapport_server
