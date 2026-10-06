---
title: ''
summary: ''
date: 2026-06-17
type: landing

sections:
  - block: dev-hero
    id: about
    content:
      username: me
      greeting: "Hi, I'm"
      show_status: false
      show_scroll_indicator: false
      typewriter:
        enable: false
        prefix: "I write about"
        strings:
          - "associative memories"
          - "attention"
          - "statistical physics"
          - "cybernetics"
          - "dynamical systems"
          - "emergent collective computational capabilities"
          - "energy-based models"
          - "entropy production"
          - "hopfield networks"
          - "ising models"
          - "many-body systems"
          - "mean-field theory"
          - "neural networks"
          - "near-equilibrium dynamics"
          - "spin systems"
          - "transformers"
          - "vector-spin models"
        type_speed: 100
        delete_speed: 50
        pause_time: 2000
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
        padding: ["1rem", "0", "0", "0"]

  - block: collection
    id: blog
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
      view: article-grid
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#ffffff"
      spacing:
        padding: ["1rem", "0", "2rem", "0"]

---
