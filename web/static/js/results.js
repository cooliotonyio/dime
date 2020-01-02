'use strict';

class Image extends React.Component {
  render () {
    return (
      <div className="col-2">
        <img 
          className="w-100" 
          src={this.props.source}
          onClick ={(event) => this.props.click_handler(this.props.i)}>
        </img>
      </div>
    )
  }
}

class LineGraph extends React.Component {

  chartRef = React.createRef();

  componentDidMount() {
    const myChartRef = this.chartRef.current.getContext("2d"); 

    const data = {
      labels: Array.from({length: this.props.data.length}, (v, k) => k+1),
      datasets: [{
        label: this.props.label,
        data: this.props.data,
        backgroundColor: "rgba(255,99,132,0.2)",
        borderColor: "rgba(255,99,132,1)",
      }]
    }

    const options = {
      maintainAspectRatio: false,
      scales: {
        yAxes: [{
          stacked: true,
          gridLines: {
            display: true,
            color: "rgba(255,99,132,0.2)"
          }
        }],
        xAxes: [{
          gridLines: {
            display: false
          }
        }]
      }
    }

    new Chart(myChartRef, {
        type: "line",
        data: data,
        options: options
    });
  }

  render() {
    return <canvas id="myChart" ref={this.chartRef}/>
  }
}

class Modal extends React.Component {
  constructor(props) {
    super(props);
    this.create_metadata = this.create_metadata.bind(this);
    this.create_actions = this.create_actions.bind(this);
    this.render_target = this.render_target.bind(this);
    this.use_as_query = this.use_as_query.bind(this);
  }

  create_metadata() {
    let results = this.props.results;
    let i = this.props.i;
    return (
      <div id="modalMetadata">
        <h2>Metadata</h2>
        <p>
          Distance: {results.dis[i]}<br/>
          Dataset name: {results.dataset_name}<br/>
          Index name: {results.index_name}<br/>
          Post processing: {results.post_processing}<br/>
          Idx: {results.idx[i]}<br/>
          Model name: {results.model_name}<br/>
          Modality: {results.index_modality}<br/>
          Target: {results.results[i]}</p>
      </div>
    )
  }

  use_as_query() {
    let results = this.props.results;
    const data = {
      target: results.idx[this.props.i],
      modality: "dataset",
      dataset_name: results.dataset_name,
      num_results: results.num_results,
      index_name: results.index_name,
    }
    window.location.href = window.location.origin + "/query?" + $.param(data);
  }

  create_actions() {
    return (
      <div id="modalActions">
        <h2>Actions</h2>
        <button className="btn btn-warning" 
          onClick={(event)=>this.props.close_modal()}>
          Close Modal</button>
        <button className="btn btn-info px-2"
          onClick={(event)=>this.use_as_query()}>
          Use as query</button>
      </div>
    )
  }

  render_target(){
    const target = this.props.results.results[this.props.i];

    if ("text" === this.props.results.index_modality) {
      return <h2>{target}</h2>
    }

    if ("image" === this.props.results.index_modality) {
      return <img src={this.props.server_url + "/" + target}/>
    }
  }

  render() {
    if (this.props.i !== null){
      return (
        <div id="modal" className="result-modal d-block">
          <div className="result-modal-body">
            <div className="row w-100 my-5 justify-content-center">
              <div className="col-8 text-center">
                {this.render_target()}
              </div>
            </div>
            <div className="row w-100 my-5 justify-content-center">
              <div className="col-3">
                {this.create_actions()}
              </div>
              <div className="col-5">
                {this.create_metadata()}
              </div>
            </div>
          </div>
        </div>
      )
    } else {
      return(<div id="modal" className="result-modal d-none"></div>)
    }
  }
}

class Header extends React.Component {
  constructor(props) {
    super(props);
    this.create_query_input = this.create_query_input.bind(this);
    this.create_tags = this.create_tags.bind(this);
    this.set_tags = this.set_tags.bind(this);
    this.state = {
      tags: ["Loading tags..."],
      modality: this.props.data.modality,
      target: this.props.data.target
    }
    console.log(props)
  }

  set_tags() {
    let form_data = new FormData();
    form_data.append("modality", this.props.data.modality)
    form_data.append("target", this.props.data.target)
    form_data.append("num_results", this.props.data.num_results)
    if (this.props.data.modality === "dataset") {
      form_data.append("dataset_name", this.props.data.dataset_name)
    }
    console.log(this.props.data)
    fetch(this.props.data.server_url + "/query", {
      method: "POST",
      body: form_data})
      .then(r => r.json())
      .then((response) => {
        this.setState({
          tags: response.results.results,
          modality: response.results.modality,
          target: response.results.target
        });
      })
  }

  componentDidMount() {
    this.set_tags();
  }

  create_query_input() {
    switch(this.state.modality) {
      case "text":
        return this.state.target;
      case "image":
        return <img height='100' src={this.props.data.server_url+"/"+this.state.target}/>;
    }  
  }

  create_tags() {
    var tags = "";
    for (var i = 0; i < this.state.tags.length; i++) {
      tags = tags + this.state.tags[i] + ", ";
    }
    return tags;
  }

  render() {
    return (
      <div id="header">
        <div className="row justify-content-center">
          <div className="col-8 text-center">
            <h1 className="pt-5">Query:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
          <div className="col-1 text-center">
            <h4>Modality:</h4> {this.props.data.modality}<br/>
            <a className="btn btn-primary pt-3" role="button" href="/">Search Another</a>
          </div>
          <div className="col-4 text-center">
            <h4>Input:</h4>
            {this.create_query_input()}
          </div>
          <div className="col-4 text-center">
            <h4>Relevant Tags:</h4>
            {this.create_tags()}
          </div>
        </div>
      </div>
    )
  }
}

class ResultsHeader extends React.Component {
  constructor(props) {
    super(props);
    this.createTabs = this.createTabs.bind(this);
    this.createGraph = this.createGraph.bind(this);
    this.handle_tab_press = this.handle_tab_press.bind(this);

    this.state = {
      available_indexes: null,
    }
  }

  createGraph(results) {
    return <LineGraph label={results.dataset_name} data={results.dis}/>
  }

  componentDidMount(){
    const data = {available_indexes: this.props.results.modality};
    const url = this.props.server_url + "/info?" + $.param(data);
    fetch(url)
      .then(result => result.json())
      .then(
        (result) => {
          this.setState({available_indexes: result.available_indexes})},
        (error) => {
          console.log(error);
          this.setState({available_indexes: []});
        }
      );
  }

  handle_tab_press(index_name) {
    const data = {
      target: this.props.results.target,
      modality: this.props.results.modality,
      num_results: this.props.results.num_results,
      index_name: index_name,
    }
    window.location.href = window.location.origin + "/query?" + $.param(data);
  }

  createTabs() {
    var tabs = [];
    if (this.state.available_indexes !== null) {
      for (let [i, index_name] of this.state.available_indexes.entries()) {
        if (index_name === this.props.results.index_name){
          tabs.push(
            <button 
              key={i}
              type="button"
              className="btn btn-primary disabled">
              {index_name}
            </button>)
        } else {
          tabs.push(
            <button 
              key={i}
              type="button"
              className="btn btn-secondary"
              onClick={(e) => this.handle_tab_press(index_name)}>
              {index_name}
            </button>)
        }
      }
    } else {
      tabs = (<div>Loading indexes...</div>)
    }
    return tabs
  }

  render() {
    return (
      <div id="resultsHeader">
        <div className="row pt-5">
          <div className="col text-center">
            <h1>Results:</h1>
          </div>
        </div>
        <div className="row justify-content-center">
        <div className="col-5 text-center">
          <div className="btn-group" role="group" aria-label="Basic example">
            {this.createTabs()}
          </div>
        </div>
        <div className="col-5">
          {this.createGraph(this.props.results)}
        </div>
      </div>
    </div>
    )
  }

}

class ResultsBody extends React.Component {
  constructor(props) {
    super(props);
    this.createResults = this.createResults.bind(this);
    this.handle_click = this.handle_click.bind(this);
  }

  handle_click(i) {
    this.props.click_handler(i)
  }

  createResults() {
    var results = []
    if ("image" === this.props.results.index_modality) {
      for (let [i, result] of this.props.results.results.entries()){
        results.push(
          <Image
            key={i}
            i={i}
            source={this.props.server_url + "/" + result}
            click_handler={this.handle_click}/>
          )
      }
    } else if ("text" === this.props.results.index_modality) {
      for (let [i, result] of this.props.results.results.entries()){
        results.push(
          <div className="col-2" 
            key={i}
            onClick={(e)=>{this.handle_click(i)}}>
            <h4>{i+1}. {result}</h4>
          </div>)
      }
    }
    return results
  }

  render() {
    return (
      <div id="resultsBody">
        <div className="row px-5 pt-5">
          {this.createResults()}
        </div>
      </div>
    )
  }
}

class Results extends React.Component{
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div id="results">
        <ResultsHeader 
          results = {this.props.results}
          server_url = {this.props.server_url}/>
        <ResultsBody 
          results = {this.props.results}
          server_url = {this.props.server_url}
          click_handler = {this.props.click_handler}/>
      </div>
    )
  }
}

class Content extends React.Component {
  constructor(props) {
    super(props);
    var data = JSON.parse(this.props.data);
    this.state = {
      modal: null,
      server_url: data.server_url,
      target: data.target,
      modality: data.modality,
      num_results: data.num_results,
      index_name: data.index_name,
      results: null,
      data: data
    };
    this.show_modal = this.show_modal.bind(this);
    this.close_modal = this.close_modal.bind(this);
    this.get_results = this.get_results.bind(this);
    this.render_results = this.render_results.bind(this);
  }

  show_modal(i){
    this.setState({
      modal: i
    });
  }

  close_modal(){
    this.setState({modal: null});
  }

  componentDidMount() {
    this.get_results(this.state.data);
  }

  get_results(data) {
    let form_data = new FormData();
    form_data.append("modality", data.modality)
    form_data.append("target", data.target)
    form_data.append("index_name", data.index_name)
    form_data.append("num_results", data.num_results)
    if (data.dataset_name) {
      form_data.append("dataset_name", data.dataset_name)
    }

    fetch(data.server_url + "/query", {
      method: "POST",
      body: form_data})
      .then(r => r.json())
      .then((response) => {
        this.setState({
          target: response.results.target,
          modality: response.results.modality,
          num_results: response.results.results.length,
          results: response.results
        });
      })
  }

  render_results() {
    if (this.state.results !== null) {
      return (
        <Results 
          results={this.state.results}
          server_url={this.state.server_url} 
          click_handler={this.show_modal}/>);
    } else {
      return (
        <div className="row w-100 py-5">
          <div className="col text-center">
            <h1>Loading results...</h1>
          </div>
        </div>
      );
    }
  }

  render_modal() {
    return (<Modal 
      i={this.state.modal}
      results={this.state.results} 
      close_modal={this.close_modal} 
      server_url={this.state.server_url}/>)
  }

  render() {
    return (
      <div>
        <Header data={this.state.data}/>
        {this.render_results()}
        {this.render_modal()}
      </div>
    )
  }
}
