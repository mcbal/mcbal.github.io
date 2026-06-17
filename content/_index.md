---
title: ''
summary: ''
date: 2026-06-17
type: landing

sections:
  - block: dev-hero
    id: about-me
    content:
      username: me
      greeting: "Hi, I'm"
      show_status: false
      show_scroll_indicator: false
      typewriter:
        enable: true
        prefix: "I write about"
        strings:
          - "machine learning"
          - "statistical physics"
          - "energy-based models"
          - "transformers"
        type_speed: 70
        delete_speed: 40
        pause_time: 2500
      cta_buttons: []
    design:
      style: centered
      avatar_shape: circle
      animations: true
      background:
        color:
          light: "#ffffff"
          dark: "#ffffff"
      spacing:
        padding: ["5rem", "0", "6rem", "0"]

  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: "Notes on attention, energy landscapes, spin systems, and neural dynamics"
      filters:
        folders:
          - blog
        exclude_featured: false
      count: 6
      order: desc
      archive:
        enable: true
        text: "Browse all posts"
        link: "/blog/"
    design:
      view: card
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#ffffff"
      spacing:
        padding: ["5rem", "0", "4rem", "0"]
---
