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
    this.text_change = this.text_change.bind(this);
    this.file_change = this.file_change.bind(this);
  }

  text_change(event){
    this.props.set_text(event.target.value);
  }

  file_change(event) {
    this.props.set_file(event.target.files[0]);
  }

  get_input() {
    var input;
    if (this.props.selected_modality == "text"){
      input = (<input className="form-control" type="text" placeholder="Enter text" onChange={this.text_change}/>);
    } else {
      input = (<input type="file" className="form-control-file" onChange={this.file_change}/>);
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
          <div className="btn-group tabs-scroll w-100" role="group" aria-label="search modalities">
            {this.render_available_indexes()}
          </div>
        </div>
      </div>
    )
  }
}

class Search extends React.Component {
  constructor(props) {
    super(props);

    this.handle_modality_change = this.handle_modality_change.bind(this);
    this.handle_selected_index_change = this.handle_selected_index_change.bind(this);
    this.handle_submit = this.handle_submit.bind(this);
    this.get_submit_button = this.get_submit_button.bind(this);
    this.set_file = this.set_file.bind(this);
    this.set_text = this.set_text.bind(this);

    let data = JSON.parse(this.props.data);
    this.state = {
      supported_modalities: data.supported_modalities,
      selected_modality: data.supported_modalities[0],
      server_url: data.server_url,
      selected_index: null,
      file: null,
      text: null,
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

  get_submit_button() {
    const active_button = (<button className="btn btn-success" onClick={this.handle_submit}>Search</button>);
    const disabled_button= (<button className="btn btn-secondary disabled">Search</button>);
    if (this.state.selected_index !== null) {
      if ("text" === this.state.selected_modality) {
        if (this.state.text !== null) {return active_button;}
      }
      if ("image" === this.state.selected_modality) {
        if (this.state.file !== null) {return active_button;}
      }
      if ("audio" === this.state.selected_modality) {
        if (this.state.file !== null) {return active_button;}
      }
      if ("video" === this.state.selected_modality) {
        if (this.state.file !== null) {return active_button;}
      }
    }
    return disabled_button;
  }

  handle_submit() {
    console.log(this.state);
    const data = {
      index_name: this.state.selected_index,
      modality: this.state.selected_modality,
      num_results: "30"
    }
    if ("text" == this.state.selected_modality) {
      data.target = this.state.text;
      this.handle_query_redirect(data);
    } else {
      let form_data = new FormData();
      form_data.append('modality', this.state.selected_modality);
      form_data.append('file', this.state.file);
      fetch(this.state.server_url + "/file_upload", {
        method: 'POST',
        body: form_data})
        .then(r => r.json())
        .then((result) => {
          if (!result.error) {
            data.target = result.target;
            this.handle_query_redirect(data);
          } else {
            console.log(result.error);
          }
        });
    }
  }

  handle_query_redirect(data) {
    window.location.href = window.location.origin + "/query?" + $.param(data)
  }

  set_file(new_file) {
    // TODO: Check if file is valid
    this.setState({
      text: null,
      file: new_file
    });
  }

  set_text(new_text) {
    //TODO: Check if text is valid
    this.setState({
      text: new_text,
      file: null
    });
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
            set_file={this.set_file}
            set_text={this.set_text}/>
          <AvailableIndexes 
            server_url={this.state.server_url}
            selected_modality={this.state.selected_modality}
            selected_index={this.state.selected_index}
            handle_selected_index_change={this.handle_selected_index_change}/>
          {this.get_submit_button()}
        </div>
      </div> 
    )
  }
}
