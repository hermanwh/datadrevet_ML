import React from "react";
import * as tf from "@tensorflow/tfjs";
import "./Download.css";
import Plot from "react-plotly.js";

class DownloadFile extends React.Component {
  state = {
    loading: false,
    loadedProject: false,
    data: [{ y: [0.1, 0.5, 0.7] }],
    layout: {
      width: 420,
      height: 340,
      padding: 0,
      title: "Data from test rigg",
      plot_bgcolor: "#fff"
    },
    frames: [],
    config: {}
  };

  getData = () => {
    return Math.random();
  };

  componentDidMount() {
    setInterval(() => {
      var k = this.state.data[0].y.slice();
      k.push(Math.random());
      k.slice(0, 1);
      console.log(k);
      this.setState({
        data: [
          {
            y: k
          }
        ]
      });
    }, 2000);
  }

  loadModel = async () => {
    try {
      const model = await tf.loadLayersModel("http://localhost:8000/model");
      console.log("Model loaded. ", model);
      model.predict(tf.ones([null, 3]), [1]);

      return model;
    } catch (err) {
      console.error(err);
    } finally {
      this.setState({
        loadedProject: true
      });
    }
  };

  render() {
    return (
      <div className="Container">
        <div className="Card">
          <div className="container">
            <div className="Title">Predefined model</div>
            <div className="ImageWrapper">
              <img alt="download_model" src="model.png" className="Image" />
            </div>
            <div className="ButtonWrapper">
              <button onClick={this.loadModel}>
                {this.state.loading
                  ? "Loading..."
                  : "Download predefined model"}
              </button>
            </div>
            <p />
          </div>
        </div>
        {this.state.loadedProject && (
          <div className="Card">
            <div className="plots">
              <div className="plot__wrapper">
                <Plot
                  data={this.state.data}
                  layout={this.state.layout}
                  frames={this.state.frames}
                  config={this.state.config}
                  onInitialized={figure => this.setState(figure)}
                  onUpdate={figure => this.setState(figure)}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }
}

export default DownloadFile;
