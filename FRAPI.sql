-- -----------------------------------------------------
-- Table `images`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `images` ;

CREATE  TABLE `images` (
  `id` INTEGER PRIMARY KEY AUTOINCREMENT ,
  `type` TEXT NOT NULL ,
  `name` TEXT ,
  `orig_img` TEXT NOT NULL UNIQUE,
  `face_img` TEXT
  ) ;
