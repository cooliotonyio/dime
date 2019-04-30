'use strict';

class Header extends React.Component {
  constructor(props) {
    super(props);
    this.create_query_input = this.create_query_input.bind(this);
    this.create_tags = this.create_tags.bind(this);
  }

  create_query_input() {
    switch(this.props.data.input_modality) {
      case "text":
        return this.props.data.query_input;
      case "image":
        return "<img height='100' src="+this.props.data.query_input+"/>";
    }  
  }

  create_tags() {
    var tags = "";
    for (var i = 0; i < this.props.data.num_datasets; i++) {
      var result = this.props.data.results[i];
      if (result.modality == "text") {
        for (var j = 0; j < result.num_results; j++) {
          tags = tags + result.data[j] + ", "
        }
      }
    }
    return tags;
  }

  render() {
    console.log("Rendering header...");
    var input_modality = this.props.data.input_modality;
    var query_input = this.create_query_input();
    var tags = this.create_tags();
    return (
      <div id="header">
        <div className="row justify-content-center">
          <div className="col-8 text-center">
            <h1 className="pt-5">Query:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
          <div className="col-1 text-center">
            <h4>Modality:</h4> {input_modality}<br/>
            <a className="btn btn-primary pt-3" role="button" href="/">Search Another</a>
          </div>
          <div className="col-4 text-center">
            <h4>Input:</h4>
            <div dangerouslySetInnerHTML={{__html: query_input}}></div>
          </div>
          <div className="col-4 text-center">
            <h4>Relevant Tags:</h4>
            {tags}
          </div>
        </div>
      </div>
    )
  }
}

class ResultsHeader extends React.Component {
  constructor(props) {
    super(props);
    this.changeDataset = this.changeDataset.bind(this);
    this.createTabs = this.createTabs.bind(this);
  }

  changeDataset(dataset){
    this.props.changeDatasetHandler(dataset);
  }

  createTabs() {
    let tabs = [];
    for (var i = 0; i < this.props.datasets.length; i++){
      let dataset = this.props.datasets[i];
      if (dataset.dataset === this.props.currentDataset.dataset) {
        var button_class = "btn btn-primary"
      } else {
        var button_class = "btn btn-secondary"
      }
      tabs.push(
        <button 
          key = {i}
          type="button" 
          className={button_class} 
          onClick ={(event)=>{this.changeDataset(dataset)}}>
          {dataset.dataset} with {dataset.model}
        </button>)
    }
    return tabs
  }

  render() {
    return (
      <div id="resultsHeader">
        <div className="row justify-content-center">
        <div className="col-3 text-center">
          <h1 className="pt-5">Results:</h1>
          <div className="btn-group" role="group" aria-label="Basic example">
            {this.createTabs()}
          </div>
        </div>
        <div className="col-6">
          <canvas id="chart"></canvas>
        </div>
      </div>
    </div>
    )
  }

}

class ResultsBody extends React.Component {
  render() {
    return (
      <div id="resultsBody">
        <h1>Body</h1>
        <h4>{JSON.stringify(this.props.value)}</h4>
      </div>
    )
  }
}
class Results extends React.Component{
  constructor(props) {
    super(props);
    for (var i = 0; i < this.props.data.num_datasets; i++) {
      var dataset = this.props.data.results[i];
      if (dataset.modality == "image") {
        this.state = {"dataset": dataset}
        break;
      }
    }

    this.changeDatasetHandler = this.changeDatasetHandler.bind(this);
  }

  changeDatasetHandler (dataset) {
    console.log(dataset);
    this.setState({
      "dataset" : dataset
    });
  }

  render() {
    console.log("Rendering results...")
    return (
      <div id="results">
        <ResultsHeader 
          datasets={this.props.data.results} 
          currentDataset={this.state.dataset}
          changeDatasetHandler={this.changeDatasetHandler}/>
        <ResultsBody value={this.state.dataset}/>
      </div>
    )
  }
}
class Content extends React.Component {
  render() {
    var data = JSON.parse(this.props.data);
    return (
      <div>
        <Header data={data}></Header>
        <Results data={data}></Results>
      </div>
    );
  }
}
