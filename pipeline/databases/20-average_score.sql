-- Stocke et calcul la moyenne d'un etudiant
DELIMITER $$

CREATE PROCEDURE ComputeAverageScoreForUser(IN uid INT)
BEGIN
	DECLARE avg_score FLOAT;

	SELECT AVG(score) INTO avg_score
	FROM corrections
	WHERE user_id = uid;

	UPDATE users
	SET average_score = avg_score
	WHERE id = uid;
END$$

DELIMITER ;
