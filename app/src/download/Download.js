import React from "react";
import * as tf from "@tensorflow/tfjs";
import "./Download.css";

class DownloadFile extends React.Component {
  state = {
    loading: false,
    loadedProject: true
  };

  loadModel = async () => {
    try {
      const model = await tf.loadLayersModel("http://localhost:8000/model");
      console.log("Model loaded. ", model);
      model.predict(tf.ones([null, 3]), [1]);

      return model;
    } catch (err) {
      console.error(err);
    }
  };

  render() {
    return (
      <div className="container">
        <div className="Title">Predefined model</div>
        <div className="ImageWrapper">
          <img alt="download_model" src="model.png" className="Image" />
        </div>
        <div className="ButtonWrapper">
          <button onClick={this.loadModel}>
            {this.state.loading ? "Loading..." : "Download predefined model"}
          </button>
        </div>
        <p />
      </div>
    );
  }
}

export default DownloadFile;
