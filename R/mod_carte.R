# =============================================================================
# mod_carte.R — Carte interactive des alertes
# =============================================================================

# Facteur d'échelle pour l'affichage de l'incidence.
# À déplacer dans global.R pour être partagé entre modules.
# Actuel : 1000 (incidence pour 1000 habitants — standard SP-Palu / SIMR).
FACTEUR_ECHELLE <- 1000

mod_carte_ui <- function(id) {
  ns <- NS(id)
  tagList(
    br(),
    bslib::layout_columns(
      col_widths = c(9, 3),
      bslib::card(
        bslib::card_header(icon("map"), " Carte des alertes par district"),
        leaflet::leafletOutput(ns("map"), height = "560px")
      ),
      bslib::card(
        bslib::card_header(icon("sliders"), " Filtres"),
        selectInput(ns("annee_sel"), "Annee :",
                    choices = 2021:2025, selected = 2025),
        selectInput(ns("mois_sel"), "Mois :",
                    choices = setNames(1:12, c("Janvier","Fevrier","Mars","Avril",
                                               "Mai","Juin","Juillet","Aout",
                                               "Septembre","Octobre","Novembre","Decembre")),
                    selected = 12),
        selectInput(ns("var_sel"), "Variable :",
                    choices = c("Niveau alerte"      = "alerte",
                                "Incidence predite"  = "pred",
                                "Incidence reelle"   = "reel")),
        br(),
        actionButton(ns("btn_carte"), "Afficher la carte",
                     icon = icon("map"), class = "btn-primary w-100"),
        br(), br(),
        tags$h6("Legende"),
        tags$div(
          tags$span(style="display:inline-block;width:14px;height:14px;background:#27AE60;border-radius:2px;margin-right:6px;"),
          "VERT - Situation normale", br(),
          tags$span(style="display:inline-block;width:14px;height:14px;background:#F7DC6F;border-radius:2px;margin-right:6px;"),
          "JAUNE - Vigilance (Q1)", br(),
          tags$span(style="display:inline-block;width:14px;height:14px;background:#F39C12;border-radius:2px;margin-right:6px;"),
          "ORANGE - Risque modere", br(),
          tags$span(style="display:inline-block;width:14px;height:14px;background:#E74C3C;border-radius:2px;margin-right:6px;"),
          "ROUGE - Epidemie potentielle", br(), br(),
          tags$span(style="display:inline-block;width:14px;height:14px;background:#CCCCCC;border-radius:2px;margin-right:6px;"),
          "Donnees manquantes"
        )
      )
    )
  )
}

mod_carte_server <- function(id, rv, shapefile_path) {
  moduleServer(id, function(input, output, session) {

    # ── CORRECTION 1 : reprojection WGS84 obligatoire pour Leaflet ──────────
    # Le shapefile est en EPSG:32630 (UTM zone 30N, coordonnées métriques).
    # Leaflet exige EPSG:4326 (degrés décimaux lon/lat).
    shp <- sf::st_read(shapefile_path, quiet = TRUE) |>
      sf::st_transform(crs = 4326)

    # ── CORRECTION 2 : supprimer Cas_palu du shapefile ───────────────────────
    # Le shapefile contient déjà une colonne Cas_palu (données statiques).
    # rv$test_alert contient aussi Cas_palu (données épidémio réelles).
    # Sans suppression, left_join crée Cas_palu.x et Cas_palu.y
    # → le popup et la palette numérique plantent silencieusement.
    shp <- shp[, c("Nom_DS", "Numero_DS", "Region", "geometry")]

    output$map <- leaflet::renderLeaflet({
      input$btn_carte

      isolate({
        if (!rv$trained || is.null(rv$test_alert)) {
          return(
            leaflet::leaflet() |>
              leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron) |>
              leaflet::setView(lng = -1.5, lat = 12.3, zoom = 7)
          )
        }

        df <- rv$test_alert
        df <- df[df$annee == as.integer(input$annee_sel) &
                   df$mois  == as.integer(input$mois_sel), ]

        # ── CORRECTION 3 : diagnostic jointure ──────────────────────────────
        # Si la carte reste grise après le fix CRS, vérifier dans la console R
        # que les noms matchent bien entre shapefile et données.
        # Décommenter les deux lignes ci-dessous pour diagnostiquer :
        # message("Shapefile Nom_DS : ", paste(head(shp$Nom_DS), collapse = ", "))
        # message("Donnees  Nom_DS  : ", paste(head(df$Nom_DS),  collapse = ", "))
        # message("Lignes df filtre : ", nrow(df))

        # ── CORRECTION 4 : harmonisation des 6 noms de districts ───────────────
        # Le shapefile utilise des noms légèrement différents des données DHIS2
        corrections_noms <- c(
          "DS Karangasso-Vigue" = "DS Karangasso Vigue",
          "DS Lena"             = "DS Léna",
          "DS N'dorola"         = "DS N'Dorola",
          "DS Seg- noghin"      = "DS Sig-Noghin",
          "DS Sig- noghin"      = "DS Sig-Noghin",
          "DS Seguenega"        = "DS Séguénéga",
          "DS Ziniare"          = "DS Ziniaré"
        )
        shp_corr        <- shp
        shp_corr$Nom_DS <- dplyr::recode(shp_corr$Nom_DS, !!!corrections_noms)
        geo_join <- dplyr::left_join(shp_corr, df, by = "Nom_DS")

        pal_alerte <- leaflet::colorFactor(
          palette  = c("#27AE60", "#F7DC6F", "#F39C12", "#E74C3C"),
          domain   = c("NORMAL", "VIGILANCE", "ALERTE", "EPIDEMIE"),
          na.color = "#CCCCCC"
        )
        pal_num <- leaflet::colorNumeric(
          "YlOrRd", domain = geo_join$Pred, na.color = "#CCCCCC"
        )

        fill_color <- if (input$var_sel == "alerte") {
          pal_alerte(geo_join$Niveau_Alerte)
        } else if (input$var_sel == "pred") {
          pal_num(geo_join$Pred)
        } else {
          pal_num(geo_join$Cas_palu)
        }

        # ── Popup enrichi au clic ────────────────────────────────────────────
        popup_txt <- paste0(
          "<div style='font-family:sans-serif;font-size:13px;min-width:180px;'>",
          "<b style='font-size:14px;'>", geo_join$Nom_DS, "</b>",
          "<hr style='margin:4px 0;border-color:#ddd;'>",
          "<b>Annee :</b> ", geo_join$annee,
          " &nbsp;|&nbsp; <b>Mois :</b> ", geo_join$mois, "<br>",
          "<b>Incidence reelle :</b> ",
          format(round(geo_join$Cas_palu * FACTEUR_ECHELLE, 2), nsmall = 2),
          " /1000 hab.<br>",
          "<b>Incidence predite :</b> ",
          format(round(geo_join$Pred      * FACTEUR_ECHELLE, 2), nsmall = 2),
          " /1000 hab.<br>",
          "<b>Alerte :</b> <span style='font-weight:bold;color:",
          dplyr::case_when(
            geo_join$Niveau_Alerte == "EPIDEMIE"  ~ "#E74C3C",
            geo_join$Niveau_Alerte == "ALERTE"    ~ "#F39C12",
            geo_join$Niveau_Alerte == "VIGILANCE" ~ "#D4AC0D",
            geo_join$Niveau_Alerte == "NORMAL"    ~ "#27AE60",
            TRUE                                  ~ "#888888"
          ),
          ";'>", geo_join$Niveau_Alerte, "</span>",
          "</div>"
        )

        # ── Labels permanents des noms de districts ──────────────────────────
        # noWrap    : empêche le retour à la ligne
        # permanent : toujours visible (pas seulement au survol)
        # direction : "center" centre le label sur le polygone
        # style     : texte petit et semi-transparent pour ne pas surcharger
        label_opts <- leaflet::labelOptions(
          noWrap    = TRUE,
          permanent = TRUE,
          direction = "center",
          textOnly  = TRUE,
          style     = list(
            "font-size"   = "9px",
            "font-weight" = "bold",
            "color"       = "#1a1a1a",
            "text-shadow" = "0px 0px 3px #ffffff, 0px 0px 3px #ffffff"
          )
        )

        leaflet::leaflet(geo_join) |>
          leaflet::addProviderTiles(leaflet::providers$CartoDB.Positron) |>
          leaflet::setView(lng = -1.5, lat = 12.3, zoom = 7) |>
          leaflet::addPolygons(
            fillColor        = fill_color,
            fillOpacity      = 0.75,
            color            = "white",
            weight           = 1,
            popup            = popup_txt,
            label            = geo_join$Nom_DS,
            labelOptions     = label_opts,
            highlightOptions = leaflet::highlightOptions(
              weight       = 2,
              color        = "#2C3E50",
              fillOpacity  = 0.9,
              bringToFront = TRUE
            )
          )
      })
    })
  })
}
