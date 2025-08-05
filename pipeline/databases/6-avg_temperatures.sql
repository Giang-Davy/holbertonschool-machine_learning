-- Afficher toute les temperatures moyennes de chaques villes
SELECT city, AVG(value) AS avg_temp FROM temperatures GROUP BY city ORDER BY avg_temp DESC;
