'use strict';

class ModalityBar extends React.Component {
  constructor(props) {
    super(props);
    this.get_buttons = this.get_buttons.bind(this);
  }

  get_buttons() {
    const modalities = []
    for (let [i, modality] of this.props.supported_modalities.entries()) {
      let tab_class_name;
      if (modality === this.props.selected_modality){
        tab_class_name = "btn btn-primary";
      } else {
        tab_class_name = "btn btn-secondary";
      }
      modalities.push(
        <button 
          key={i} 
          type="button" 
          className={tab_class_name}
          onClick={(e) => this.props.handle_modality_change(modality)}
        >{modality}</button>
      );
    }
    return modalities
  }
  
  render() {
    return (
      <div className="row justify-content-center w-100">
        <div className="col-6 text-center">
          <div className="btn-group" role="group" aria-label="search modalities">
            {this.get_buttons()}
          </div>
        </div>
      </div>
    )
  }
}

class SearchEntry extends React.Component {
  constructor(props) {
    super(props);
    this.get_input = this.get_input.bind(this);
  }

  get_input() {
    var input;
    if (this.props.selected_modality == "text"){
      input = (<input className="form-control" type="text" placeholder="Enter text"/>);
    } else {
      input = (<input type="file" className="form-control-file"/>);
    }
    return input;
  }

  render() {
    return( 
      <div className="row justify-content-center w-100 pt-5">
        <div className="col-6 text-center">
          <div className="form-group">
            {this.get_input()}
          </div>
        </div>
      </div>
    )
  }
}

class AvailableIndexes extends React.Component{
  constructor(props) {
    super(props);
    this.render_available_indexes = this.render_available_indexes.bind(this);
    this.update_available_indexes = this.update_available_indexes.bind(this);
    this.state = {
      available_indexes: [],
    };
  }

  update_available_indexes() {
    const data = {available_indexes: this.props.selected_modality};
    const url = this.props.server_url + "/info?" + $.param(data);
    fetch(url)
      .then(result => result.json())
      .then(
        (result) => {this.setState({available_indexes: result.available_indexes})},
        (error) => {
          console.log(error);
          this.setState({available_indexes: []});
        }
      );
  }
  
  componentDidMount() {
    this.update_available_indexes();
  }

  componentDidUpdate(prevProps) {
    if (prevProps.selected_modality != this.props.selected_modality){
      this.update_available_indexes();
    }
  }
  
  render_available_indexes() {
    var indexes = [];
    if (this.state.available_indexes.length == 0) {
      indexes = (<div className="text-danger">No indexes are available for this modality</div>);
    } else {
      for (let [i, index_name] of this.state.available_indexes.entries()) {
        let tab_class_name;
        if (index_name === this.props.selected_index){
          tab_class_name = "btn btn-primary";
        } else {
          tab_class_name = "btn btn-secondary";
        }
        indexes.push(<button 
          key={i} 
          type="button" 
          className={tab_class_name}
          onClick={(e) => this.props.handle_selected_index_change(index_name)}
        >{index_name}</button>);
      }
    }
    return indexes;
  }

  render() {
    return (
      <div className="row justify-content-center w-100 py-5">
        <div className="col-6 text-center">
          <h2>Available indexes for {this.props.selected_modality}</h2>
          <div className="btn-group" role="group" aria-label="search modalities">
            {this.render_available_indexes()}
          </div>
        </div>
      </div>
    )
  }
}

class SubmitSearch extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    if (this.props.selected_index !== null) {
      return (<button className="btn btn-success">Search</button>);
    } else {
      return (<button className="btn btn-secondary disabled">Search</button>);
    }
  }
}

class Search extends React.Component {
  constructor(props) {
    super(props);
    this.handle_modality_change = this.handle_modality_change.bind(this);
    this.handle_selected_index_change = this.handle_selected_index_change.bind(this);
    let data = JSON.parse(this.props.data);
    this.state = {
      supported_modalities: data.supported_modalities,
      selected_modality: data.supported_modalities[0],
      server_url: data.server_url,
      selected_index: null,
    };
  }

  handle_modality_change(modality) {
    this.setState({
      selected_modality: modality,
      selected_index: null});
  }

  handle_selected_index_change(index_name){
    this.setState({selected_index: index_name});
  }

  render() {
    return (
      <div className="row justify-content-center w-100 pt-5">
        <div className="col-8 text-center">
          <h1 className="pt-5">DIME Search Engine</h1>
          <ModalityBar 
            supported_modalities={this.state.supported_modalities} 
            selected_modality={this.state.selected_modality}
            handle_modality_change={this.handle_modality_change}/>
          <SearchEntry
            selected_modality={this.state.selected_modality}
          />
          <AvailableIndexes 
            server_url={this.state.server_url}
            selected_modality={this.state.selected_modality}
            selected_index={this.state.selected_index}
            handle_selected_index_change={this.handle_selected_index_change}/>
          <SubmitSearch
            selected_index={this.state.selected_index}/>
        </div>
      </div> 
    )
  }
}
