const express = require("express");
const upload = require("./upload");
const cors = require("cors");

const server = express();

var corsOptions = {
  origin: "*",
  optionsSuccessStatus: 200
};

server.use(cors(corsOptions));

server.use(
  express.static("public")
); /* this line tells Express to use the public folder as our static folder from which we can serve static files*/

server.post("/upload", upload);

server.get("/model", function(req, res) {
  // res.sendFile('model.json')
  res.sendFile("model.json", { root: "public" });
});

server.listen(8000, () => {
  console.log("Server started!");
});
