# =============================================================================
# mod_alertes.R — Alertes mail automatiques
# =============================================================================

# Facteur d'échelle pour l'affichage de l'incidence.
# À déplacer dans global.R pour être partagé entre modules.
# Actuel : 1000 (incidence pour 1000 habitants — standard SP-Palu / SIMR).
FACTEUR_ECHELLE <- 1000

mod_alertes_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),
    bslib::layout_columns(
      col_widths = c(5, 7),
      
      bslib::card(
        bslib::card_header(icon("cog"), " Configuration SMTP"),
        textInput(ns("smtp_host"), "Serveur SMTP :", value = "smtp.gmail.com"),
        numericInput(ns("smtp_port"), "Port :", value = 587),
        textInput(ns("smtp_user"), "Email expéditeur :",
                  placeholder = "votre@email.com"),
        passwordInput(ns("smtp_pass"), "Mot de passe / App password :"),
        hr(),
        tags$h6(icon("users"), " Destinataires par district"),
        tags$p("Format — une ligne par district : ",
               tags$code("NomDistrict:email@exemple.com")),
        textAreaInput(
          ns("recipients"), NULL,
          placeholder = "DS Ouagadougou:dso_ouaga@sante.bf\nDS Bobo-Dioulasso:dso_bobo@sante.bf",
          rows = 7
        ),
        hr(),
        selectInput(
          ns("seuil_envoi"), "Envoyer alertes pour :",
          choices = c("ROUGE uniquement" = "EPIDEMIE",
                      "ORANGE et ROUGE"  = "ALERTE",
                      "Tous niveaux"     = "TOUS")
        ),
        actionButton(ns("btn_test_mail"), "Tester la connexion",
                     icon = icon("vial"), class = "btn-info btn-sm"),
        br(), br(),
        actionButton(ns("btn_send"), "Envoyer les alertes",
                     icon = icon("paper-plane"), class = "btn-danger w-100")
      ),
      
      tagList(
        bslib::card(
          bslib::card_header(icon("eye"), " Aperçu du mail d'alerte"),
          selectInput(ns("district_preview"), "Choisir un district :",
                      choices = NULL),
          uiOutput(ns("mail_preview"))
        ),
        br(),
        bslib::card(
          bslib::card_header(icon("history"), " Journal des envois"),
          DT::DTOutput(ns("tbl_logs"))
        )
      )
    )
  )
}

mod_alertes_server <- function(id, rv) {
  moduleServer(id, function(input, output, session) {
    
    logs <- reactiveVal(data.frame(
      Horodatage = character(), District = character(),
      Email      = character(), Niveau   = character(),
      Statut     = character(), stringsAsFactors = FALSE
    ))
    
    # Test de connexion SMTP
    observeEvent(input$btn_test_mail, {
      req(input$smtp_host, input$smtp_user, input$smtp_pass)
      tryCatch({
        serveur <- emayili::server(
          host     = input$smtp_host,
          port     = as.integer(input$smtp_port),
          username = input$smtp_user,
          password = input$smtp_pass,
          insecure = FALSE
        )
        msg <- emayili::envelope() |>
          emayili::from(input$smtp_user) |>
          emayili::to(input$smtp_user) |>
          emayili::subject("[TEST] Connexion SMTP — Système Alerte Paludisme BF") |>
          emayili::html("<p>Test de connexion réussi — Système d'alerte paludisme.</p>")
        serveur(msg)
        showNotification("✅ Connexion SMTP réussie — email de test envoyé !",
                         type = "message", duration = 5)
      }, error = function(e) {
        showNotification(paste0("❌ Échec connexion SMTP : ", e$message),
                         type = "error", duration = 8)
      })
    })

    # Mise à jour dynamique de la liste des districts
    observe({
      req(rv$trained, rv$future_pred)
      fp <- rv$future_pred
      districts <- unique(fp$Nom_DS)
      updateSelectInput(session, "district_preview",
                        choices = districts)
    })
    
    # Aperçu du mail
    output$mail_preview <- renderUI({
      if (!rv$trained || is.null(rv$future_pred)) {
        return(tags$p(icon("info-circle"),
                      " Lancez d'abord l'analyse pour voir l'aperçu."))
      }
      
      req(input$district_preview)
      fp <- rv$future_pred
      
      ds_data <- fp[fp$Nom_DS == input$district_preview, ]
      
      if (nrow(ds_data) == 0) {
        return(tags$p(" Aucune donnée pour ce district."))
      }
      
      max_alerte <- if ("EPIDEMIE"   %in% ds_data$Niveau_Alerte) "EPIDEMIE"
      else if ("ALERTE"    %in% ds_data$Niveau_Alerte) "ALERTE"
      else if ("VIGILANCE" %in% ds_data$Niveau_Alerte) "VIGILANCE"
      else "NORMAL"
      
      color_alerte <- switch(max_alerte,
                             EPIDEMIE  = "#E74C3C",
                             ALERTE    = "#F39C12",
                             VIGILANCE = "#F7DC6F",
                             NORMAL    = "#27AE60")
      
      bg_alerte <- switch(max_alerte,
                          EPIDEMIE  = "#FADBD8",
                          ALERTE    = "#FAD7A0",
                          VIGILANCE = "#FFFDE7",
                          NORMAL    = "#D5F5E3")
      
      # Prendre le mois avec le niveau le plus élevé comme référence
      ordre_niveaux <- c("EPIDEMIE" = 4, "ALERTE" = 3, "VIGILANCE" = 2, "NORMAL" = 1)
      ds_data$score <- ordre_niveaux[ds_data$Niveau_Alerte]
      ex <- ds_data[which.max(ds_data$score), ]

      # Tableau des 6 mois coloré
      lignes_mois <- lapply(seq_len(nrow(ds_data)), function(i) {
        row      <- ds_data[i, ]
        niv      <- row$Niveau_Alerte
        col_txt  <- switch(niv, EPIDEMIE = "#E74C3C", ALERTE = "#F39C12", VIGILANCE = "#D4AC0D", NORMAL = "#27AE60")
        col_fond <- switch(niv, EPIDEMIE = "#FADBD8", ALERTE = "#FAD7A0", VIGILANCE = "#FFFDE7", NORMAL = "#D5F5E3")
        icone    <- switch(niv,
                           EPIDEMIE  = "🔴",
                           ALERTE    = "🟠",
                           VIGILANCE = "🟡",
                           NORMAL    = "🟢")
        tags$tr(
          style = paste0("background:", col_fond, "; color:#2C3E50;"),
          tags$td(style = "padding:6px 10px; font-weight:bold;",
                  paste0(row$annee, "/M", sprintf("%02d", row$mois))),
          tags$td(style = "padding:6px 10px; text-align:right;",
                  format(round(row$Pred_ML * FACTEUR_ECHELLE, 2), nsmall = 2)),
          tags$td(style = "padding:6px 10px; text-align:right;",
                  if ("Seuil_Alerte" %in% names(row))
                    format(round(row$Seuil_Alerte * FACTEUR_ECHELLE, 2), nsmall = 2)
                  else "—"),
          tags$td(style = "padding:6px 10px; text-align:right;",
                  if ("Seuil_Epidemie" %in% names(row))
                    format(round(row$Seuil_Epidemie * FACTEUR_ECHELLE, 2), nsmall = 2)
                  else "—"),
          tags$td(style = paste0("padding:6px 10px; font-weight:bold; color:", col_txt, "; text-align:center;"),
                  paste0(icone, " ", niv))
        )
      })

      tags$div(
        class = "border rounded p-3",
        style = "background:white; border: 1px solid #dee2e6;",
        tags$h5(
          icon("exclamation-triangle"),
          paste0(" ", max_alerte, " — ", input$district_preview),
          style = paste0("color:", color_alerte, ";")
        ),
        tags$p(tags$strong("Objet : "),
               paste0("[ALERTE PALUDISME] ", input$district_preview,
                      " — Niveau max : ", max_alerte)),
        tags$hr(),
        tags$p(
          "Le système de prédiction signale les niveaux suivants pour ",
          tags$strong(input$district_preview), " sur les 6 prochains mois :"
        ),
        tags$table(
          style = "width:100%; border-collapse:collapse; font-size:13px;",
          tags$thead(
            tags$tr(
              style = "background:#2C3E50; color:white;",
              tags$th(style = "padding:6px 10px;", "Mois"),
              tags$th(style = "padding:6px 10px; text-align:right;", "Incidence prédite (/1000 hab.)"),
              tags$th(style = "padding:6px 10px; text-align:right;", "Seuil alerte (/1000 hab.)"),
              tags$th(style = "padding:6px 10px; text-align:right;", "Seuil épidémie (/1000 hab.)"),
              tags$th(style = "padding:6px 10px; text-align:center;", "Niveau")
            )
          ),
          tags$tbody(lignes_mois)
        ),
        tags$hr(),
        tags$p(
          tags$strong("Actions recommandées :"), br(),
          if (max_alerte == "EPIDEMIE") tagList(
            "• Mobiliser les équipes de riposte", br(),
            "• Vérifier les stocks ACT et TDR", br(),
            "• Renforcer la surveillance active"
          ) else if (max_alerte == "ALERTE") tagList(
            "• Renforcer la surveillance de routine", br(),
            "• Vérifier les stocks préventifs"
          ) else if (max_alerte == "VIGILANCE") tagList(
            "• Surveillance renforcée — activité au-dessus de Q1", br(),
            "• Vérifier les stocks préventifs en anticipation"
          ) else tagList(
            "• Maintenir la surveillance standard"
          )
        ),
        tags$em("Message généré automatiquement — Système d'Alerte Paludisme BF")
      )
    })
    
    # Envoi des alertes
    observeEvent(input$btn_send, {
      req(rv$trained, rv$future_pred, input$smtp_user, input$smtp_pass, input$recipients)
      
      lines <- strsplit(trimws(input$recipients), "\n")[[1]]
      lines <- lines[nzchar(trimws(lines))]
      
      recipients_map <- Filter(Negate(is.null), lapply(lines, function(l) {
        parts <- strsplit(trimws(l), ":")[[1]]
        if (length(parts) >= 2)
          list(district = trimws(parts[1]), email = trimws(parts[2]))
        else NULL
      }))
      
      if (length(recipients_map) == 0) {
        showNotification("❌ Aucun destinataire valide.", type = "error")
        return()
      }
      
      fp         <- rv$future_pred
      seuil_env  <- input$seuil_envoi
      new_logs   <- logs()
      nb_envoyes <- 0
      
      tryCatch({
        for (rec in recipients_map) {
          ds_data <- fp[fp$Nom_DS == rec$district, ]
          if (nrow(ds_data) == 0) next
          
          max_alerte <- if ("EPIDEMIE"  %in% ds_data$Niveau_Alerte) "EPIDEMIE"
          else if ("ALERTE" %in% ds_data$Niveau_Alerte) "ALERTE"
          else "NORMAL"
          
          envoyer <- (seuil_env == "EPIDEMIE"  && max_alerte == "EPIDEMIE") ||
            (seuil_env == "ALERTE"    && max_alerte %in% c("EPIDEMIE", "ALERTE")) ||
            (seuil_env == "VIGILANCE" && max_alerte %in% c("EPIDEMIE", "ALERTE", "VIGILANCE")) ||
            (seuil_env == "TOUS")
          
          if (!envoyer) next
          
          color_alerte <- switch(max_alerte,
                                 EPIDEMIE  = "#E74C3C",
                                 ALERTE    = "#F39C12",
                                 VIGILANCE = "#D4AC0D",
                                 NORMAL    = "#27AE60")
          
          bg_alerte <- switch(max_alerte,
                              EPIDEMIE  = "#FADBD8",
                              ALERTE    = "#FAD7A0",
                              VIGILANCE = "#FFFDE7",
                              NORMAL    = "#D5F5E3")
          
          # Prendre le mois avec le niveau le plus élevé comme référence
          ordre_niveaux <- c("EPIDEMIE" = 4, "ALERTE" = 3, "VIGILANCE" = 2, "NORMAL" = 1)
          ds_data$score <- ordre_niveaux[ds_data$Niveau_Alerte]
          ex <- ds_data[which.max(ds_data$score), ]

          serveur <- emayili::server(
            host     = input$smtp_host,
            port     = as.integer(input$smtp_port),
            username = input$smtp_user,
            password = input$smtp_pass,
            insecure = FALSE
          )
          
          msg <- emayili::envelope() |>
            emayili::from(input$smtp_user) |>
            emayili::to(rec$email) |>
            emayili::subject(
              paste0("[ALERTE PALUDISME] ", rec$district, " — Niveau ", max_alerte)
            ) |>
            emayili::html(paste0(
              "<div style='font-family:Arial,sans-serif; padding:20px;'>",
              
              "<h2 style='color:", color_alerte, ";'>",
              "⚠️ ALERTE ", max_alerte, " — ", rec$district,
              "</h2>",
              
              "<p><strong>Objet :</strong> [ALERTE PALUDISME] ", rec$district,
              " — Risque épidémique prédit pour ",
              ex$annee, "/M", sprintf("%02d", ex$mois), "</p>",
              
              "<hr/>",
              
              "<p>Bonjour,</p>",
              "<p>Le système de prédiction signale une situation de niveau ",
              "<strong style='color:", color_alerte, ";'>", max_alerte, "</strong>",
              " pour le district sanitaire de <strong>", rec$district, "</strong>.</p>",
              
              "<p>",
              "<strong>📅 Période :</strong> ", ex$annee, ", Mois ", ex$mois, "<br/>",
              "<strong>📊 Incidence prédite :</strong> ",
              format(round(as.numeric(ex[["Pred_ML"]]) * FACTEUR_ECHELLE, 2), nsmall = 2),
              " /1000 hab.<br/>",
              "<strong>🚨 Seuil d'alerte (Q2) :</strong> ",
              format(round(as.numeric(ex[["Seuil_Alerte"]]) * FACTEUR_ECHELLE, 2), nsmall = 2),
              " /1000 hab.<br/>",
              "<strong>🔴 Seuil épidémique (Q3) :</strong> ",
              format(round(as.numeric(ex[["Seuil_Epidemie"]]) * FACTEUR_ECHELLE, 2), nsmall = 2),
              " /1000 hab.",
              "</p>",
              
              "<p><strong>Actions recommandées :</strong><br/>",
              if (max_alerte == "EPIDEMIE") paste0(
                "• Mobiliser les équipes de riposte<br/>",
                "• Vérifier les stocks ACT et TDR<br/>",
                "• Renforcer la surveillance active"
              ) else if (max_alerte == "ALERTE") paste0(
                "• Renforcer la surveillance de routine<br/>",
                "• Vérifier les stocks préventifs"
              ) else if (max_alerte == "VIGILANCE") paste0(
                "• Surveillance renforcée — activité au-dessus de Q1<br/>",
                "• Vérifier les stocks préventifs en anticipation"
              ) else paste0(
                "• Maintenir la surveillance standard"
              ),
              "</p>",
              
              "<br/><hr/>",
              "<em>Message généré automatiquement — Système d'Alerte Paludisme BF</em>",
              "</div>"
            ))
          
          serveur(msg)
          nb_envoyes <- nb_envoyes + 1
          
          new_logs <- rbind(new_logs, data.frame(
            Horodatage = format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
            District   = rec$district,
            Email      = rec$email,
            Niveau     = max_alerte,
            Statut     = "✅ Envoyé",
            stringsAsFactors = FALSE
          ))
        }
        
        logs(new_logs)
        showNotification(
          paste0("✅ ", nb_envoyes, " alerte(s) envoyée(s) !"),
          type = "message", duration = 5
        )
        
      }, error = function(e) {
        showNotification(paste0("❌ Erreur SMTP : ", e$message),
                         type = "error", duration = 10)
      })
    })
    
    output$tbl_logs <- DT::renderDT({
      DT::datatable(
        logs(), rownames = FALSE,
        options = list(pageLength = 10, order = list(list(0, "desc")))
      )
    })
  })
}