<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
	<meta name="viewport" content="width=device-width, initial-scale=1">
	<title>TEST - HTTP & HTTPS</title>
	<style type="text/css">
		body {
			text-align: center;
		}
	</style>
</head>
<body>
	<h1>TEST - HTTP & HTTPS</h1>
	<div>
		<?php
			if (!empty($_SERVER['HTTPS']) && ('on' == $_SERVER['HTTPS'])) {
				$uri = 'https://';
				echo "目前是 HTTPS";
			} else {
				$uri = 'http://';
				echo "目前是 HTTP";
			}
			$uri .= $_SERVER['HTTP_HOST'];
			phpinfo();
		?>	
	</div>
</body>
</html>