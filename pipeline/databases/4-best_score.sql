-- afficher seulement oÙ le score est egal ou plus de 10 et affiche score en premiere colonne
SELECT score, name FROM second_table WHERE score >= 10 ORDER BY score DESC;
