-- ajoute une correction pour un etudiant
DELIMITER //

CREATE PROCEDURE AddBonus(IN user_id INT, IN project_name VARCHAR(255), IN score INT)
BEGIN
	DECLARE project_id INT;

	-- Vérifie si le projet existe déjà
	SELECT id INTO project_id FROM projects WHERE name = project_name LIMIT 1;

	-- Si le projet n'existe pas, on le crée
	IF project_id IS NULL THEN
		INSERT INTO projects (name) VALUES (project_name);
		SET project_id = LAST_INSERT_ID();
	END IF;

	-- Ajoute la correction
	INSERT INTO corrections (user_id, project_id, score)
	VALUES (user_id, project_id, score);
END;
//

DELIMITER ;
