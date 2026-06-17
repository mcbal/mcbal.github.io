---
title: ''
summary: ''
date: 2026-06-17
type: landing

sections:
  - block: collection
    id: posts
    content:
      title: Recent Posts
      subtitle: "Notes on attention, energy landscapes, spin systems, and neural dynamics"
      filters:
        folders:
          - posts
        exclude_featured: false
      count: 5
      order: desc
      archive:
        enable: true
        text: "Browse all posts"
        link: "/posts/"
    design:
      view: card
      columns: 3
      background:
        color:
          light: "#ffffff"
          dark: "#ffffff"
      spacing:
        padding: ["3rem", "0", "2rem", "0"]

  - block: dev-hero
    id: about
    content:
      username: me
      greeting: "Hi, I'm"
      show_status: false
      show_scroll_indicator: false
      typewriter:
        enable: true
        prefix: "I write about"
        strings:
          - "artificial intelligence"
          - "associative memories"
          - "attention"
          - "machine learning"
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
        padding: ["3rem", "0", "2rem", "0"]

---
