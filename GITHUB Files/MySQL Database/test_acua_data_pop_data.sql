-- MySQL dump 10.13  Distrib 8.0.27, for Win64 (x86_64)
--
-- Host: 127.0.0.1    Database: test_acua_data
-- ------------------------------------------------------
-- Server version	8.0.27

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `pop_data`
--

DROP TABLE IF EXISTS `pop_data`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pop_data` (
  `Town_ID` int NOT NULL,
  `Location` varchar(25) DEFAULT NULL,
  `Population` int NOT NULL,
  `Area_sqmi` decimal(12,2) DEFAULT NULL,
  `Pop_Density` decimal(65,30) DEFAULT NULL,
  `Source` varchar(255) DEFAULT NULL,
  `Recorded_date` datetime DEFAULT NULL,
  `Reported_pop/sqmi` decimal(12,2) DEFAULT NULL,
  `Date` int NOT NULL,
  PRIMARY KEY (`Town_ID`),
  KEY `Population_index` (`Population`),
  KEY `Date_index` (`Date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `pop_data`
--

LOCK TABLES `pop_data` WRITE;
/*!40000 ALTER TABLE `pop_data` DISABLE KEYS */;
INSERT INTO `pop_data` VALUES (1,'Egg Harbor Twp',42249,66.60,634.369369369370000000000000000000,'https://www.census.gov/quickfacts/fact/table/eggharbortownshipatlanticcountynewjersey,atlanticcountynewjersey/PST040219','2019-07-01 00:00:00',650.50,2010),(2,'Hamilton Twp',25746,111.13,231.674615315396000000000000000000,'https://www.census.gov/quickfacts/fact/table/hamiltontownshipatlanticcountynewjersey,atlanticcountynewjersey/PST040219','2019-07-01 00:00:00',238.50,2010),(3,'Galloway Twp',35618,89.07,399.887728752666000000000000000000,'https://www.census.gov/quickfacts/fact/table/gallowaytownshipatlanticcountynewjersey,atlanticcountynewjersey/PST040219','2019-07-01 00:00:00',419.30,2010),(4,'Absecon',8818,5.40,1632.962962962960000000000000000000,'https://www.census.gov/quickfacts/fact/table/abseconcitynewjersey/PST040219','2019-07-01 00:00:00',1558.70,2010),(5,'Pleasantville',20149,5.69,3541.124780316340000000000000000000,'https://www.census.gov/quickfacts/fact/table/pleasantvillecitynewjersey,abseconcitynewjersey/PST040219','2019-07-01 00:00:00',3556.20,2010),(6,'Linwood',6658,3.87,1720.413436692510000000000000000000,'https://www.census.gov/quickfacts/fact/table/linwoodcitynewjersey/PST040219','2019-07-01 00:00:00',1834.90,2010),(7,'Brigantine',8650,6.39,1353.677621283260000000000000000000,'https://www.census.gov/quickfacts/fact/table/brigantinecitynewjersey/PST040219','2019-07-01 00:00:00',1479.60,2010),(8,'Atlantic City',37743,10.75,3510.976744186050000000000000000000,'https://www.census.gov/quickfacts/fact/table/atlanticcitycitynewjersey,brigantinecitynewjersey/PST040219','2019-07-01 00:00:00',3680.80,2010),(9,'Margate',5865,1.42,4130.281690140840000000000000000000,'https://www.census.gov/quickfacts/fact/table/margatecitycitynewjersey,brigantinecitynewjersey/PST040219','2019-07-01 00:00:00',4490.50,2010),(10,'Northfield',8031,3.40,2362.058823529410000000000000000000,'https://www.census.gov/quickfacts/fact/table/northfieldcitynewjersey,margatecitycitynewjersey,brigantinecitynewjersey/PST040219','2019-07-01 00:00:00',2533.50,2010),(11,'Somers Point',10174,4.03,2524.565756823820000000000000000000,'https://www.census.gov/quickfacts/fact/table/somerspointcitynewjersey/PST040219','2019-07-01 00:00:00',2678.70,2010),(12,'Ventnor Township',10650,1.57,6783.439490445860000000000000000000,'https://www.census.gov/quickfacts/fact/table/ventnorcitycitynewjersey,hamiltontownshipatlanticcountynewjersey,atlanticcountynewjersey/POP060210','2019-02-23 00:00:00',5457.40,2010),(13,'Costal Alternative',0,1.00,0.000000000000000000000000000000,'Estimated','2019-02-23 00:00:00',0.00,2021);
/*!40000 ALTER TABLE `pop_data` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2022-06-03 15:19:36
